# Checkpoint: bounce_scorer
**Created**: 2026-02-19 15:10
**Status**: completed

## Summary
Built a historical bounce backscanner (`scanners/bounce_backscanner.py`) that scans ALL tickers across date ranges for capitulation bounce setups using Polygon's bulk daily endpoint. Also updated `bounce_scorer.py` to V2 criteria (removed vol_expansion, added pct_change_3 and pct_off_52wk_high). The backscanner was validated against known trades across multiple crash periods (Aug 2024, May 2022, Apr 2025).

## What Was Done
- Created `scanners/bounce_backscanner.py` (~500 lines) — historical bulk bounce scanner
  - Copies infrastructure from `historical_backscanner.py` (bulk fetch, ticker index, pickle cache, trading calendar, CLI)
  - Bounce-specific pre-filters: consecutive down days >= 2, gap down, 5%+ off 30d high
  - Computes all bounce metrics locally from cached daily OHLCV (zero API calls per ticker)
  - Scores via `BouncePretrade.validate()` from `analyzers/bounce_scorer.py`
  - ETF detection via `KNOWN_ETFS` from `bounce_trader.py` + dollar-volume heuristic
  - `--verify` mode checks recall against `bounce_data.csv` with miss diagnosis
  - `--min-bounce` filter (default 2%) confirms actual low-to-close bounce before saving
  - Added `bounce_low_to_close` metric to output (measures real intraday opportunity vs open-to-close)
- `analyzers/bounce_scorer.py` was updated to V2 by user before this session:
  - Removed `vol_expansion` (rho=0.04, zero predictive power)
  - Added `pct_change_3` (rho=-0.700, #2 predictor) and `pct_off_52wk_high` (rho=-0.487)
  - Now 7 pre-trade criteria (was 6), 8 historical criteria (was 7)
  - GO threshold: >= 6/7, CAUTION: 5/7, NO-GO: <= 4/7

## Key Decisions & Findings
- **Default cap filter is `Large,ETF,Medium`** — many tradeable names (COIN, SOXL, SMCI, CRWD) land in Medium by dollar-volume heuristic
- **Default min-bounce is 2%** (low-to-close) — filters out setups where the stock never actually bounced
- **Premarket RVOL unavailable** from daily aggs, but V2 removed vol_expansion so this is no longer a limitation
- **bounce_low_to_close** is the real measure of opportunity — NVDA Jul 25 was -0.7% open-to-close but +5.6% low-to-close
- Pre-filter of consecutive_down_days >= 2 means single-day plunges (IntradayCapitch) are excluded by design (MSTR May 2022 miss)

## Validation Results
| Period | Matches | GO | Known Recall | Notes |
|--------|---------|-----|--------------|-------|
| Jul-Aug 2024 | 374 | 31 | 4/4 (100%) | Aug 5 Japan carry trade crash: 196 candidates, 15 GO |
| May 2022 | 202 | 28 | 4/5 (80%) | LUNA/UST crash. MSTR missed (1 down day) |
| Apr 2025 | 603 | 59 | 3/3 (100%) | Tariff crash. Apr 7: 299 candidates, 38 GO, 14 A+ |

## Current State
- Scanner is fully functional and tested across 3 crash periods
- Cached data exists for Jul-Aug 2024, May 2022, May 2023, Apr 2025
- Results CSVs saved in `data/` directory

## Next Steps
- Consider running full multi-year scan (e.g. 2022-2025) for comprehensive bounce dataset
- Could add bounce_low_to_close and bounce_day_return to the verification comparison
- Could study which GO trades were actually profitable (next-day follow-through analysis)
- Could feed confirmed bounces back into bounce_data.csv to expand training set
- Consider adding `--setup-type` filter (weakstock only, strongstock only)

## Key Files
| File | Role |
|------|------|
| `scanners/bounce_backscanner.py` | **NEW** — Historical bulk bounce scanner |
| `analyzers/bounce_scorer.py` | V2 bounce scoring (7 pre-trade criteria) — imported by backscanner |
| `scanners/bounce_trader.py` | Live bounce monitor — provides `KNOWN_ETFS` set |
| `scanners/historical_backscanner.py` | Template — reversal backscanner (infrastructure copied from here) |
| `data/bounce_data.csv` | 54 historical bounce trades for `--verify` mode |
| `data/backscanner_cache/` | Pickle cache of fetched grouped daily data |
