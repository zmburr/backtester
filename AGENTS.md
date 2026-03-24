# Backtester Project

Algorithmic trading analysis and backtesting platform for equities. Analyzes stock price movements, volume patterns, and technical breakouts to test reversal, momentum, and bounce trading strategies on historical data.

## Project Structure

```
backtester/
├── backtesters/                    # Core backtesting execution
│   └── main_backtester.py          # Main entry point for backtests
├── backtesting_strategies/         # Trading signal and exit logic
│   ├── entry_signals.py            # Entry signal identification (4 types)
│   ├── exit_signals.py             # Multi-bar exit strategies
│   ├── stop_strategies.py          # Stop loss placement logic
│   └── trade.py                    # Trade object class
├── data_collectors/                # Data aggregation and feature engineering
│   ├── combined_data_collection.py # Main data pipeline for breakout/reversal analysis
│   ├── bounce_collector.py         # Bounce strategy data enrichment (50+ metrics)
│   └── strong_bounce_collector.py  # Scanner feature extractor (SHEL/Trillium)
├── data_queries/                   # External data source integration
│   ├── polygon_queries.py          # Polygon.io REST API wrapper (primary)
│   └── trillium_queries.py         # SHEL DataGateway integration (fallback)
├── analyzers/                      # Analysis and visualization
│   ├── charter.py                  # Chart generation (Plotly & mplfinance)
│   ├── bounce_scorer.py            # Pre-trade checklist & setup classification
│   ├── bounce_exit_targets.py      # Exit target framework for bounces
│   ├── reversal_scorer.py          # Parabolic short scoring (6 cap-adjusted criteria)
│   └── reversal_pretrade.py        # Per-setup-type reversal pre-trade validator
├── scanners/                       # Stock screening and monitoring
│   ├── setup_screener.py           # Universe screener for parabolic short + bounce setups
│   ├── historical_backscanner.py   # Historical backscanner (all tickers, bulk Polygon data)
│   ├── stock_screener.py           # Watchlist screener with percentile rankings
│   ├── live_watcher.py             # Live market monitoring
│   └── bounce_trader.py            # Live bounce trade monitor with TTS alerts
├── scripts/
│   └── generate_report.py          # Daily trading report generation
├── support/
│   ├── config.py                   # Email, API keys, environment variables
│   └── llm_client.py               # LLM provider routing
├── data/                           # CSV data storage
│   ├── breakout_data.csv           # Momentum/breakout trading signals
│   ├── reversal_data.csv           # Reversal trading signals
│   ├── bounce_data.csv             # Bounce trading signals
│   └── bounce_scored.csv           # Bounce trades with scoring results
└── charts/                         # Generated chart outputs
```

## Key Concepts

### Trade Types
- **Momentum/Breakout** (`breakout_data.csv`): Long positions on upward breakouts
- **Reversal** (`reversal_data.csv`): Short positions on overextended bounces
- **Bounce** (`bounce_data.csv`): Long positions on capitulation/selloff reversals

### Entry Signals (4 types)
1. Premarket Low Break - Price breaks below premarket low, then reverses
2. Premarket High Break - Price breaks above premarket high, then reverses
3. Open Price Break - Price breaks outside open price, then reverses
4. 2-Minute Break - Price breaks previous bar's high/low

### Exit Strategies
- **Quick Strategy**: Exit immediately when price breaks prior bar low/high
- **Delayed Strategy**: Wait for the bar to close before exiting (trailing bar strategy)

---

## Bounce Strategy

The bounce strategy trades long positions on capitulation selloffs. It's the most fully-developed strategy with live monitoring, pre-trade validation, and data-driven exit targets.

### Setup Types (Auto-Classified)

Classification is based on position relative to 50MA and 30-day momentum:

| Setup Type | Classification | Win Rate | Avg P&L | Sample |
|------------|----------------|----------|---------|--------|
| **GapFade_weakstock** | Below 50MA + negative 30d momentum | 75% | +13.3% | 16 A-grade |
| **GapFade_strongstock** | Above 50MA + flat/positive 30d momentum | 71% | +2.7% | 24 A-grade |

### Pre-Trade Checklist (5 Criteria)

Each criterion returns GO / CAUTION / NO-GO based on cap-specific thresholds:

1. **Deep selloff** - Total % decline from recent high
2. **Consecutive down days** - 2-4 depending on cap
3. **Discount from 30d high** - 14-50% depending on cap
4. **Gap down capitulation** - 1-10% gap depending on cap
5. **Volume signal** - Prior day RVOL OR premarket RVOL exceeds threshold

**Bonus signals**: Closed outside lower Bollinger Band, prior day near lows, deep off 52wk high
**Warning signals**: SPY strong (+2%), IntradayCapitch pattern (17% WR, -13.6% avg)

### Exit Targets by Cap

Based on historical hit rates from 36 trades:

| Cap | T1 (Hit Rate) | T2 (Hit Rate) | T3 (Hit Rate) |
|-----|---------------|---------------|---------------|
| ETF | 0.5x ATR (71%) | 1.0x ATR (71%) | Gap Fill (29%) |
| Medium | 0.5x ATR (71%) | Gap Fill (50%) | 1.0x ATR (58%) |
| Small | 0.5x ATR (50%) | 1.0x ATR (50%) | Gap Fill (67%) |

**Dip Risk**: Avg dip before bounce is -14% (1.68 ATRs); max observed -25% to -30%

### Bounce Intensity Score

0-100 composite score based on percentile ranking of:
- Selloff depth
- Gap down %
- Consecutive down days
- Volume climax

### Key Bounce Files

| File | Purpose |
|------|---------|
| `scanners/bounce_trader.py` | Live monitor with real-time TTS alerts |
| `analyzers/bounce_scorer.py` | Pre-trade checklist & setup classification |
| `analyzers/bounce_exit_targets.py` | ATR-based exit framework |
| `data_collectors/bounce_collector.py` | Data enrichment (50+ metrics) |
| `data/bounce_data.csv` | Historical bounce trades |

---

## Data Flow

```
CSV (ticker + date) → combined_data_collection.py → Enriched DataFrame
                                ↓
                      main_backtester.py
                                ↓
              entry_signals → stop_strategies → exit_signals
                                ↓
                      Results CSV with risk metrics
```

## Important Files

### `data_queries/polygon_queries.py`
Primary data source. Key functions:
- `get_intraday(ticker, date, multiplier, timespan)` - Fetch candlestick data
- `get_daily(ticker, date)` - OHLC for specific date
- `get_levels_data(ticker, date, window, multiplier, timespan)` - Historical OHLCV
- `get_atr(ticker, date)` - Average True Range (14-day)
- `get_ticker_mavs_open(ticker, date)` - Moving average distances (uses high of day)
- `fetch_and_calculate_volumes(ticker, date)` - Volume metrics across timeframes

### `data_collectors/combined_data_collection.py`
Main data pipeline. Enriches trade data with:
- Volume metrics (premarket, 5/10/15/30-min intervals)
- Moving average distances (9 EMA, 10/20/50/200 SMA)
- ATR distance from 50-day MA
- Range expansion metrics
- Historical volume comparisons

Fill functions are defined in dictionaries (`fill_functions_momentum`, `fill_functions_reversal`) that map column names to enrichment functions.

### `backtesting_strategies/trade.py`
Core trade object storing:
- Entry signals and prices
- Stop loss strategies
- Drawdown tracking
- Risk metrics (max loss, max profit, R:R ratio)
- Position size: 1000 units (long) or -1000 (short)

### `scanners/bounce_trader.py`
Live market monitor for bounce setups. Features:
- Event-driven threading with TTS audio alerts
- Real-time bar-5s data stream from Trillium/SHEL DataGateway
- Automatic market cap classification and setup type detection
- **Price alerts**: ATR-based targets (0.5x, 1.0x, 1.5x, 2.0x, 2.7x ATR), gap fill levels, selloff retrace targets
- **Time alerts**: 10:00 AM (early low check), 2:30 PM (scale reminder), 3:30 PM (overnight hold), 3:55 PM (EOD summary)
- Pre-trade scoring integration via BouncePretrade

Key classes:
- `BounceTradeManager` - Core monitor class
- `BounceDataAdapter` - Trillium stream wrapper
- `BounceContext` - Pre-computed setup data
- `compute_bounce_intensity()` - 0-100 percentile scoring

### `analyzers/bounce_scorer.py`
Pre-trade validation and historical scoring. Features:
- Auto-classifies stocks into GapFade_weakstock or GapFade_strongstock
- Validates 5 pre-trade criteria with GO/CAUTION/NO-GO status
- Fetches live metrics for real-time validation
- Cap-specific thresholds for each criterion

Key classes:
- `BouncePretrade` - Live pre-trade validator
- `BounceScorer` - Historical trade scorer
- `SetupProfile` - Profile dataclass with thresholds
- `ChecklistResult` - Validation output

### `scanners/setup_screener.py`
Universe screener that searches across tickers for parabolic short and capitulation bounce setups. Fetches daily OHLCV from Polygon (single API call per ticker), computes all metrics locally, then scores against criteria.

**Parabolic Short screening** — tries typed classification first (e.g. 3DGapFade via `ReversalPretrade`), then falls back to generic `ReversalScorer` thresholds with a directional gate (`pct_from_9ema > 0`):
1. % above 9EMA (per-cap)
2. Prior day range vs ATR (range expansion)
3. RVOL (volume expansion)
4. Consecutive up days
5. Gap up %
- Typed setups require score >= 4 (GO), generic keeps >= 3

**Capitulation Bounce screening** (6 pre-trade criteria from `bounce_scorer.py`):
1. Selloff depth (total % decline)
2. Consecutive down days
3. % off 30-day high
4. Gap down %
5. Prior day range vs ATR
6. Volume signal (prior day RVOL or premarket RVOL)

Key classes:
- `SetupScreener` - Main screener with `screen_universe()`, `screen_ticker()`, `get_candidates()`
- `ScreenResult` - Per-ticker result with scores, grades, recommendations, and `parabolic_setup_type`
- Pre-built ticker universes: `MEGA_CAP`, `MOMENTUM_NAMES`, `ETFS`, `MINERS_COMMODITIES`, `SMALL_MICRO`
- `build_universe()` - Combine and deduplicate ticker lists
- `get_polygon_tickers()` - Pull active tickers from Polygon reference API

### `scanners/historical_backscanner.py`
Historical backscanner that scans ALL tickers across date ranges for reversal setup types. Uses Polygon's `get_grouped_daily_aggs()` endpoint for massive efficiency: 1 API call per date returns OHLCV for all ~12,000 tickers.

**Pipeline**: Bulk fetch -> build per-ticker history -> compute metrics locally -> classify + score
- ~560 API calls for 1 year scan (vs 250,000 per-ticker calls with the old approach)
- Filters: price > $5, avg volume > 500K (reduces ~12K tickers to ~2,800 screenable)
- Caches fetched data to pickle for resume capability

Key classes:
- `HistoricalBackscanner` - Main scanner with `fetch_market_data()`, `scan_date()`, `scan_range()`
- Output CSV: date, ticker, cap, setup_type, score, grade, recommendation, and all criteria values

### `analyzers/reversal_scorer.py`
Scores parabolic short setups based on 6 cap-adjusted criteria. Returns score (0-6), grade (A+/A/B/C/F), and GO/NO-GO recommendation. Used as the **generic fallback** by `setup_screener.py` when no typed setup is detected.

Key classes:
- `ReversalScorer` - Scorer with cap-specific thresholds
- `CriteriaThresholds` - Threshold dataclass per cap size
- `CAP_THRESHOLDS` - Dict of thresholds for Micro/Small/Medium/Large/ETF

### `analyzers/reversal_pretrade.py`
Per-setup-type reversal pre-trade validator. Mirrors the `BouncePretrade` pattern but for reversal (parabolic short) setups. Uses per-setup, per-cap thresholds derived from historical Grade A+B trades instead of generic one-size-fits-all thresholds.

Starting setup type: **3DGapFade** (2+ euphoric up days + gap up on fade day)
- Classification: `consecutive_up_days >= 2` AND `gap_pct > 0` AND `pct_from_9ema > 0.04`
- 33 historical trades (23 A, 8 B) — GO trades: 90% WR, +16% avg P&L
- Per-cap thresholds for 5 criteria: pct_from_9ema, range/ATR, RVOL, up days, gap %

Key classes/functions:
- `ReversalPretrade` - Pre-trade validator with `validate()` and `print_checklist()`
- `ReversalSetupProfile` - Per-cap threshold dataclass
- `classify_reversal_setup(metrics)` - Auto-detects setup type from metrics
- `REVERSAL_SETUP_PROFILES` - Dict of profiles keyed by setup type

### `data_collectors/bounce_collector.py`
Data enrichment pipeline for bounce trades. Computes 50+ metrics:
- **Bounce day stats**: gap%, open-to-high%, open-to-close%, open-to-low%
- **Selloff context**: consecutive down days, % off 30d/52wk high, selloff total %
- **Bollinger bands**: lower/upper distance, position, width, days since upper band
- **Volume climax**: 3-day trend, direction, down/up day ratio
- **Intraday timing**: time of low (bucketed), HOD-to-LOD duration, bounce duration
- **ATR metrics**: ATR %, ATR move, range expansion, % from MVAs

## CSV Column Order (reversal_data.csv)

The reversal CSV must maintain this column order:
```
date, ticker, trade_grade, cap, intraday_setup, setup, atr_pct, atr_pct_move,
avg_daily_vol, breaks_ath, breaks_fifty_two_wk, close_at_lows, close_green_red,
day_of_range_pct, gap_pct, hit_green_red, hit_prior_day_hilo, move_together,
one_day_before_range_pct, pct_change_120, pct_change_15, pct_change_3, pct_change_30,
pct_change_90, pct_from_10mav, pct_from_200mav, pct_from_20mav, pct_from_50mav,
atr_distance_from_50mav, percent_of_premarket_vol, percent_of_vol_in_first_10_min,
percent_of_vol_in_first_15_min, percent_of_vol_in_first_30_min, percent_of_vol_in_first_5_min,
percent_of_vol_on_breakout_day, percent_of_vol_one_day_before, percent_of_vol_three_day_before,
percent_of_vol_two_day_before, premarket_vol, reversal_duration, reversal_open_close_pct,
reversal_open_low_pct, reversal_open_post_low_pct, reversal_open_to_day_after_open_pct,
spy_open_close_pct, three_day_before_range_pct, time_of_high_price, time_of_low,
time_of_reversal, two_day_before_range_pct, vol_in_first_10_min, vol_in_first_15_min,
vol_in_first_30_min, vol_in_first_5_min, vol_on_breakout_day, vol_one_day_before,
vol_three_day_before, vol_two_day_before, bp, npl, size
```

## Date Formats

- CSV input dates: `%m/%d/%Y` (e.g., "01/15/2024")
- API/internal dates: `%Y-%m-%d` (e.g., "2024-01-15")
- Functions convert between formats as needed

## Common Commands

```bash
# Run backtester
python backtesters/main_backtester.py

# Fill missing data in CSVs
python data_collectors/combined_data_collection.py

# Generate daily report
python scripts/generate_report.py
# or
run_generate_report.bat

# Run stock screener (percentile ranking)
python scanners/stock_screener.py

# Run setup screener (parabolic short + bounce universe scan)
python scanners/setup_screener.py                # today, default universe
python scanners/setup_screener.py 2025-01-15     # specific date

# --- Bounce Strategy Commands ---

# Live bounce trade monitoring (real-time with TTS alerts)
python scanners/bounce_trader.py NVDA Medium
python scanners/bounce_trader.py COIN              # auto-detect cap
python scanners/bounce_trader.py NVDA Medium --dry-run

# Score historical bounce trades
python analyzers/bounce_scorer.py

# Fill bounce_data.csv with missing metrics
python data_collectors/bounce_collector.py

# --- Reversal Pre-Trade Commands ---

# Validate reversal thresholds against reversal_data.csv
python analyzers/reversal_pretrade.py

# --- Historical Backscanner Commands ---

# Scan 1 year for 3DGapFade setups
python -m scanners.historical_backscanner --start 2024-01-01 --end 2024-12-31 --setup 3DGapFade

# Scan 5 years
python -m scanners.historical_backscanner --start 2020-01-01 --end 2024-12-31 --setup 3DGapFade
```

## Dependencies

- **Data**: polygon-api-client, pandas, numpy, pandas-market-calendars
- **Visualization**: plotly, mplfinance, matplotlib
- **ML/LLM**: openai, groq, together (via support/llm_client.py)

## Notes

- Polygon.io is primary data source; Trillium/SHEL is fallback
- All timestamps are Eastern time (US/Eastern)
- ATR calculations adapt window size for IPOs with limited history
- Moving average calculations use `timestamp` parameter to get historical values (not current)
