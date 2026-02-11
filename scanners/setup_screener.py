"""
Stock Setup Screener

Screens a universe of tickers for two setup types:
1. Parabolic Short - stocks extended to the upside, ripe for reversal
2. Capitulation Bounce - stocks capitulating after extended selloff, ripe for bounce

Computes all screening metrics from a single batch of daily OHLCV data per ticker
(one API call), then applies the same cap-adjusted criteria used by ReversalScorer
and BouncePretrade.

Pre-trade screening criteria:
  Parabolic Short (5 of 6 reversal criteria - excludes reversal_pct which is the trade result):
    1. % above 9EMA (cap-adjusted)
    2. Prior day range vs ATR (range expansion)
    3. RVOL (volume expansion)
    4. Consecutive up days
    5. Gap up %

  Capitulation Bounce (6 pre-trade criteria - excludes bounce_pct which is the trade result):
    1. Selloff depth (total % decline)
    2. Consecutive down days
    3. % off 30-day high
    4. Gap down %
    5. Prior day range vs ATR (range expansion)
    6. Volume signal (prior day RVOL or premarket RVOL)

Usage:
    from scanners.setup_screener import SetupScreener

    screener = SetupScreener()
    results = screener.screen_universe(['AAPL', 'TSLA', 'NVDA'], '2025-01-15')
    screener.print_results(results)

    # Screen with per-ticker market cap overrides
    caps = {'AAPL': 'Large', 'TSLA': 'Large', 'NVDA': 'Large'}
    results = screener.screen_universe(['AAPL', 'TSLA', 'NVDA'], '2025-01-15', ticker_caps=caps)

    # Filter to only candidates
    parabolic = screener.get_candidates(results, setup_type='parabolic')
    bounces = screener.get_candidates(results, setup_type='bounce')

    # Screen with pre-computed metrics (no API calls)
    metrics = {'pct_from_9ema': 0.45, 'prior_day_range_atr': 2.1, ...}
    result = screener.screen_ticker_from_metrics('NVDA', metrics, cap='Large')
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tabulate import tabulate

from data_queries.polygon_queries import (
    get_levels_data,
    get_daily,
    adjust_date_to_market,
    fetch_and_calculate_volumes,
)
from analyzers.reversal_scorer import ReversalScorer, CAP_THRESHOLDS as REVERSAL_THRESHOLDS
from analyzers.bounce_scorer import (
    BouncePretrade,
    classify_stock,
    SETUP_PROFILES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Pre-built ticker universes for convenience
# ---------------------------------------------------------------------------

MEGA_CAP = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'AVGO', 'LLY', 'JPM', 'V', 'UNH', 'MA', 'XOM', 'COST', 'HD',
            'PG', 'JNJ', 'ABBV', 'WMT', 'NFLX', 'CRM', 'BAC', 'ORCL']

MOMENTUM_NAMES = ['NVDA', 'TSLA', 'AVGO', 'PLTR', 'APP', 'CRWD', 'IONQ', 'RGTI',
                  'RKLB', 'OKLO', 'SMR', 'CRDO', 'NBIS', 'HUBS', 'DUOL', 'MU',
                  'AMD', 'MSTR', 'COIN', 'HOOD', 'ARM', 'SMCI', 'ANET']

ETFS = ['SPY', 'QQQ', 'IWM', 'DIA', 'ARKK', 'XBI', 'XLF', 'XLE', 'XLK',
        'GLD', 'SLV', 'GDXJ', 'IBIT', 'ETHE', 'TLT', 'HYG']

MINERS_COMMODITIES = ['GOLD', 'NEM', 'PAAS', 'HL', 'AG', 'SLV', 'GLD', 'GDXJ',
                      'MP', 'VALE', 'FCX', 'CLF', 'X', 'AA']

SMALL_MICRO = ['HYMC', 'BITF', 'IREN', 'QS', 'BE', 'OPEN', 'CRML', 'BETR',
               'PL', 'USAR', 'CRWV', 'FIG', 'FIGR']


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

    Fetches daily OHLCV data from Polygon (single API call per ticker),
    computes all needed metrics locally, then scores against cap-adjusted
    criteria derived from historical Grade A trades.
    """

    def __init__(self):
        self.reversal_scorer = ReversalScorer()
        self.bounce_checker = BouncePretrade()

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_metrics(self, ticker: str, date: str) -> Dict:
        """
        Compute all screening metrics from daily OHLCV data.

        Fetches ~310 calendar days of daily bars in a single API call, then
        computes locally:
          - 9 EMA, 50/200 SMA and % distance from each
          - 14-day ATR and prior day range as multiple of ATR
          - Consecutive up days and consecutive down days
          - RVOL (prior day volume / 20-day avg volume)
          - Gap % (today open vs prior close)
          - % changes over 3/15/30/90/120 day windows
          - % off 30-day high and 52-week high
          - Selloff depth and start price
          - Bollinger Band positioning
          - Prior day close vs high/low %

        Args:
            ticker: Stock ticker symbol
            date: Date string in YYYY-MM-DD format

        Returns:
            Dictionary of computed metrics, or empty dict if no data
        """
        metrics = {}

        # Single API call: ~310 calendar days -> ~200+ trading days
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

        # hist = completed bars only (exclude today if present)
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

            # Prior day range as multiple of ATR
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
        metrics['prior_day_rvol'] = metrics['rvol_score']  # alias for bounce checker

        # Premarket RVOL not available from daily bars (requires intraday data)
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
    # Parabolic short scoring (5 pre-trade criteria)
    # ------------------------------------------------------------------

    def _score_parabolic_short(self, metrics: Dict, cap: str) -> Tuple[int, int, str, str, Dict]:
        """
        Score pre-trade parabolic short criteria.

        Uses 5 of the 6 ReversalScorer criteria (excludes reversal_pct which
        is the trade result, not a pre-trade condition).

        Returns:
            (score, max_score, grade, recommendation, criteria_details)
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

    # ------------------------------------------------------------------
    # Capitulation bounce scoring (6 pre-trade criteria)
    # ------------------------------------------------------------------

    def _score_capitulation_bounce(self, ticker: str, metrics: Dict,
                                   cap: str) -> Tuple[int, int, str, str, Dict]:
        """
        Score pre-trade capitulation bounce criteria using BouncePretrade.

        Auto-classifies the stock as weakstock/strongstock based on positioning
        relative to the 200-day MA, then applies the matching profile thresholds.

        Returns:
            (score, max_score, setup_type, recommendation, criteria_details)
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

        Fetches data from Polygon, computes metrics, and scores against both
        setup types.

        Args:
            ticker: Stock ticker symbol
            date: Date string YYYY-MM-DD
            cap: Market cap category (Micro, Small, Medium, Large, ETF)

        Returns:
            ScreenResult with scores for both setup types
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

        Use this when you already have the metrics from your own data source
        or want to test screening logic without Polygon.

        Args:
            ticker: Stock ticker symbol
            metrics: Dictionary of pre-computed metrics (see compute_metrics
                     for required keys)
            cap: Market cap category

        Returns:
            ScreenResult with scores for both setup types
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
            ticker=ticker,
            cap=cap,
            parabolic_score=p_score,
            parabolic_max_score=p_max,
            parabolic_grade=p_grade,
            parabolic_recommendation=p_rec,
            parabolic_criteria=p_criteria,
            bounce_score=b_score,
            bounce_max_score=b_max,
            bounce_setup_type=b_setup,
            bounce_recommendation=b_rec,
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
    # Universe screening
    # ------------------------------------------------------------------

    def screen_universe(self, tickers: List[str], date: str,
                        cap: str = 'Medium',
                        ticker_caps: Optional[Dict[str, str]] = None) -> List[ScreenResult]:
        """
        Screen a list of tickers for both setup types.

        Args:
            tickers: List of ticker symbols to screen
            date: Date to screen (YYYY-MM-DD)
            cap: Default market cap category for all tickers
            ticker_caps: Optional dict mapping ticker -> cap to override default
                         e.g. {'AAPL': 'Large', 'HYMC': 'Micro', 'SPY': 'ETF'}

        Returns:
            List of ScreenResult objects (one per ticker)
        """
        results = []
        total = len(tickers)

        print(f"\nScreening {total} tickers for setups on {date}...")
        print(f"Default cap: {cap}")
        if ticker_caps:
            print(f"Per-ticker cap overrides: {len(ticker_caps)} tickers")
        print("-" * 60)

        for i, ticker in enumerate(tickers, 1):
            ticker_cap = ticker_caps.get(ticker, cap) if ticker_caps else cap
            logging.info(f"[{i}/{total}] Screening {ticker} ({ticker_cap})...")

            result = self.screen_ticker(ticker, date, ticker_cap)
            results.append(result)

            # Progress indicator
            if result.error:
                print(f"  [{i}/{total}] {ticker}: ERROR - {result.error}")
            elif result.is_parabolic_candidate or result.is_bounce_candidate:
                flags = []
                if result.is_parabolic_candidate:
                    flags.append(f"PARABOLIC {result.parabolic_score}/{result.parabolic_max_score}")
                if result.is_bounce_candidate:
                    flags.append(f"BOUNCE {result.bounce_score}/{result.bounce_max_score}")
                print(f"  [{i}/{total}] {ticker}: ** {' | '.join(flags)} **")
            else:
                print(f"  [{i}/{total}] {ticker}: no setup detected")

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
            results: List of ScreenResult from screen_universe
            setup_type: 'parabolic', 'bounce', or 'both'
            min_parabolic_score: Minimum parabolic short score (default 3/5)
            min_bounce_score: Minimum bounce score (default 4/6)

        Returns:
            Filtered list of ScreenResult
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
        """
        Print screening results as formatted tables.

        Args:
            results: List of ScreenResult
            show_all: If True, show all tickers. If False, show only candidates.
        """
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
              f"{len([r for r in valid if r.is_parabolic_candidate])} parabolic candidates | "
              f"{len([r for r in valid if r.is_bounce_candidate])} bounce candidates | "
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

        # Key metrics
        print(f"\nKEY METRICS:")
        print(f"  Price: ${m.get('current_price', 'N/A'):.2f}" if m.get('current_price') else "  Price: N/A")
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
# Ticker universe helpers
# ---------------------------------------------------------------------------

def get_polygon_tickers(market: str = 'stocks', ticker_type: str = 'CS',
                        exchange: str = None, limit: int = 500) -> List[str]:
    """
    Fetch active tickers from Polygon.io reference API.

    Args:
        market: Market type ('stocks', 'otc', 'crypto')
        ticker_type: Ticker type ('CS' = common stock, 'ETF', 'ADRC', etc.)
        exchange: Exchange filter (e.g. 'XNAS' for NASDAQ, 'XNYS' for NYSE)
        limit: Maximum tickers to return

    Returns:
        List of ticker symbols
    """
    from polygon.rest import RESTClient
    client = RESTClient(api_key="pcwUY7TnSF66nYAPIBCApPMyVrXTckJY")

    tickers = []
    try:
        for t in client.list_tickers(market=market, type=ticker_type,
                                     exchange=exchange, active=True,
                                     limit=limit, order='asc'):
            tickers.append(t.ticker)
            if len(tickers) >= limit:
                break
    except Exception as e:
        logging.error(f"Error fetching tickers from Polygon: {e}")

    return tickers


def build_universe(*groups) -> List[str]:
    """
    Build a deduplicated ticker list from pre-built groups.

    Args:
        *groups: Any combination of pre-built lists (MEGA_CAP, MOMENTUM_NAMES,
                 ETFS, MINERS_COMMODITIES, SMALL_MICRO) or custom lists

    Returns:
        Deduplicated list of ticker symbols

    Example:
        universe = build_universe(MEGA_CAP, MOMENTUM_NAMES, ['CUSTOM1', 'CUSTOM2'])
    """
    seen = set()
    result = []
    for group in groups:
        for ticker in group:
            if ticker not in seen:
                seen.add(ticker)
                result.append(ticker)
    return result


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

    # Default to today
    today = datetime.now()
    if today.weekday() == 5:
        adjusted_date = today - timedelta(days=1)
    elif today.weekday() == 6:
        adjusted_date = today - timedelta(days=2)
    else:
        adjusted_date = today
    date = adjusted_date.strftime('%Y-%m-%d')

    # Parse optional CLI args
    if len(sys.argv) > 1:
        date = sys.argv[1]

    # Build universe from pre-built lists
    universe = build_universe(MOMENTUM_NAMES, MEGA_CAP, ETFS)

    # Optional: per-ticker cap overrides
    ticker_caps = {}
    for t in MEGA_CAP:
        ticker_caps[t] = 'Large'
    for t in ETFS:
        ticker_caps[t] = 'ETF'
    for t in SMALL_MICRO:
        ticker_caps[t] = 'Small'

    print(f"\nSetup Screener - {date}")
    print(f"Universe: {len(universe)} tickers")
    print(f"Looking for: Parabolic Short + Capitulation Bounce setups\n")

    screener = SetupScreener()
    results = screener.screen_universe(universe, date, cap='Medium',
                                       ticker_caps=ticker_caps)

    # Print summary tables
    screener.print_results(results)

    # Print detailed reports for top candidates
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
