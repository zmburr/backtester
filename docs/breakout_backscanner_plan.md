# Breakout Backscanner — Implementation Plan

Mirror `scanners/bounce_backscanner.py`. Scan history (or daily) for tickers
with setup conditions similar to A-grade breakouts in `breakout_data.csv`,
ranked by the existing `compute_breakout_intensity` against the A+B reference
distribution.

## Architecture

```
scanners/breakout_backscanner.py

class BreakoutBackscanner:
    def __init__(self, start_date, end_date, cap_filter=None,
                 min_intensity=None):
        # min_intensity defaults to BreakoutScorer's tradable_min per profile

    def fetch_market_data(self):
        # Bulk-fetch grouped daily aggs from Polygon (1 call/date for ~12K tickers)
        # Cache to data/breakout_backscanner_cache/<date>.parquet

    def scan_date(self, date):
        for each ticker on this date:
            # 1. PRE-FILTER (cheap, rejects ~90% before any feature computation):
            #    - price >= $5
            #    - 20d avg vol >= 500K
            #    - above all of 10/20/50 MA  (extension exists)
            #    - within 15% of 52wk high   (proximity to trigger)
            #    - ATR % >= 2.5%             (volatility regime)
            # 2. COMPUTE FEATURES from local history (no API calls — derive
            #    from the bulk daily DataFrame accumulated to that point):
            #    - mirror combined_data_collection's forward-looking features
            # 3. CLASSIFY setup_type via derive_setup_type
            # 4. SCORE via BreakoutPretrade.validate() + compute_breakout_intensity()
            # 5. KEEP if intensity_tier in {FULL_SIZE, REDUCED_SIZE}

    def scan_range(self, start, end):
        # Iterate dates, accumulate results, emit sorted CSV
```

## Pre-filter design (efficiency)

`get_grouped_daily_aggs(date)` returns OHLCV for every ticker on one date.
With caching, scanning 1 year ~= 250 API calls.

The 5-step pre-filter reduces ~12K tickers to ~100–500 candidates per date,
on which the full feature suite runs.

## Output schema

```csv
date,ticker,cap,setup_type,pretrade_score,pretrade_max,intensity,intensity_tier,
  pct_from_9ema,pct_to_52wk_high,atr_pct,gap_pct,...
  breakout_open_high_pct,close_at_highs    # forward-looking outcome (historical mode)
```

Two modes:
- **Historical** — scan past dates, include actual outcomes for backtest validation
- **Live** — scan today, no outcomes; surfaces actionable candidates

## Pattern-tag filter

`--pattern <tag>` flag (e.g., `--pattern 2DayContinuation`) to filter results
to a specific pattern family. Useful for surfacing historical D1+D2 sequence
starts, gap_n_go candidates, etc.

## Important gotchas

1. **No forward-leakage on `pct_to_52wk_high`** — must compute using only
   data BEFORE the scan date. Bulk fetch runs sequentially with growing history.
2. **News context unavailable** — backscanner can't know if a stock had news
   on a date. Live use leaves `t = NaN` → no news_dayX tag → setups score
   as "pure pattern" breakouts. User manually flags news context after the
   fact via the watchlist.
3. **Small reference distribution** — A+B reference is currently only 39 trades.
   Intensity scores will be noisy until more trades accumulate. Refresh
   thresholds quarterly.

## Validation plan

1. **Backtest mode**: scan Jan-Mar 2024 (~5 known A-grade trades occurred
   then) and confirm those exact ticker-dates appear in the FULL_SIZE bucket.
2. **Coverage**: how many additional FULL_SIZE candidates surface that the
   user didn't trade? Are any big winners that were missed?
3. **False positives**: any FULL_SIZE candidate that flopped (open-to-high < 2%)?

## Build phases

| Phase | Scope | Notes |
|---|---|---|
| **A** | Bulk fetch + pre-filter + feature computation (no scoring) | new file, ~300 lines |
| **B** | Hook in BreakoutPretrade + intensity, output CSV | ~100 line extension |
| **C** | Pattern-tag filter, --start/--end CLI, --cap filter | small additions |
| **D** | Validation — backtest Jan-Mar 2024, surface metrics | small script |

A+B as one PR; C+D iteratively after.

## Open questions to resolve before building

1. **Cap filter default** — scan all caps, or default to `Mid+Large+ETF`
   (where the user's A trades cluster)?
2. **Universe** — full Polygon universe (~12K, lots of micro junk) or
   filter to a curated universe?
3. **`pct_to_52wk_high` pre-filter cutoff** — `-10%` strict or `-15%` loose?
4. **Priority report integration** — should backscanner output feed
   `priority_report.py` as additional candidates beyond the manual watchlist,
   or stay standalone?

## Reference

- Mirrors `scanners/bounce_backscanner.py` architecture
- Uses `analyzers/breakout_scorer.py` (BreakoutPretrade, compute_breakout_intensity,
  classify_breakout_setup, _INTENSITY_THRESHOLDS)
- Uses `data_collectors/combined_data_collection.py` (derive_setup_type,
  feature computation logic — port locally to avoid per-ticker API calls)
