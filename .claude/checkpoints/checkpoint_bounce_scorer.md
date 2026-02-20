# Checkpoint: bounce_scorer
**Created**: 2026-02-19 17:30
**Updated**: 2026-02-20 12:15
**Status**: completed

## Summary
Built a historical bounce backscanner, expanded the training set from 54 to 123 trades across 4 crash periods, then visually reviewed all new trades via a Streamlit chart review tool, removed 30 low-quality trades (down to 93), and re-updated all hardcoded statistics across 4 files. All updates verified working.

## What Was Done

### Phase 1: Expansion (Feb 19)
- Created `scanners/bounce_backscanner.py` (~500 lines) — historical bulk bounce scanner
  - Copies infrastructure from `historical_backscanner.py` (bulk fetch, ticker index, pickle cache, trading calendar, CLI)
  - Bounce-specific pre-filters: consecutive down days >= 2, gap down, 5%+ off 30d high
  - Computes all bounce metrics locally from cached daily OHLCV (zero API calls per ticker)
  - Scores via `BouncePretrade.validate()` from `analyzers/bounce_scorer.py`
  - ETF detection via `KNOWN_ETFS` from `bounce_trader.py` + dollar-volume heuristic
  - `--verify` mode checks recall against `bounce_data.csv` with miss diagnosis
  - `--min-bounce` filter (default 2%) confirms actual low-to-close bounce before saving
- Expanded `data/bounce_data.csv` from 54 to 123 trades:
  - 6 A+ trades from 2024 (RIOT, HUT, DPST, NVDA Jul 25, MSTR, RGTI)
  - 63 real stock + major ETF trades across 4 crash periods (Aug 2024, May 2022, Apr 2025, Oct 2022)
  - Excluded all leveraged/inverse ETPs (SOXL, TQQQ, TSLL, MSTU, NVDL, etc.)
  - Ran `bounce_collector.py` to fill all 92 metric columns
- Created `scripts/compute_bounce_stats.py` — generates comprehensive stats JSON
- Updated all 4 stat files for 123-trade dataset

### Phase 2: Visual Review & Cleanup (Feb 20)
- Created `scripts/bounce_chart_review.py` — Streamlit visual review tool
  - Side-by-side daily (1Y context with MAs) + intraday (5min with VWAP) charts
  - Navigation with prev/next buttons + dropdown selector
  - Key metrics in sidebar (gap%, down days, ATR%, RVOL, Bollinger, etc.)
  - Flag-for-removal checkbox with JSON persistence (`data/bounce_removals.json`)
  - Apply removals button to delete flagged trades from CSV
- Visually reviewed all 69 new trades, flagged 30 for removal
- Grade corrections: NVDA 7/25/2024 A→B, MSTR 12/20/2024 A→B
- Dataset trimmed from 123 to **93 trades**
- Recomputed all stats via `scripts/compute_93_stats.py`
- **Re-updated all 4 files** for 93-trade dataset:
  - `analyzers/bounce_scorer.py`: Weakstock 21 A-grade/95% WR/+11.2%, Strongstock 15 A-grade/94% WR/+12.0%, all reference medians updated
  - `analyzers/bounce_exit_targets.py`: Medium n=50 (was 75), Large n=19 (was 24), hit rates updated
  - `scanners/bounce_trader.py`: Intensity spec V4 with stronger correlations, fixed score denominator bug (was "out of 6" → "out of 7"), late low WR 32% (was 45%)
  - `scripts/generate_report.py`: All stats, cheat sheet, profile tables, correlations, cluster/overnight stats updated

## Key Decisions & Findings
- **Removing 30 weak trades strengthened all signals**: correlations improved significantly
- **Correlations stronger after cleanup**: selloff rho -0.451 (was -0.337 at 123, -0.712 at 54)
- **Both profiles improved dramatically**: weakstock 95% WR (was 86%), strongstock 94% WR (was 83%)
- **GO trades now 95.6% WR, +14.8% avg** (was 93.4%, +12.2% at 123 trades)
- **Dip risk**: median -0.42 ATR below open before bounce
- **Gap fill rates**: 86% half fill, 63% full fill
- **Leveraged/inverse ETPs excluded** from training set
- **Score denominator bug fixed** in bounce_trader.py (was saying "out of 6", corrected to "out of 7")

## Scoring Results (93 trades, 8 criteria)
| Score | Trades | Win Rate | Avg P&L |
|-------|--------|----------|---------|
| 8/8 | 33 | 100.0% | +17.2% |
| 7/8 | 27 | 92.6% | +9.6% |
| 6/8 | 13 | 92.3% | +2.7% |
| 5/8 | 7 | 71.4% | +0.3% |
| <=4/8 | 13 | 15.4% | -5.2% |

**GO (7-8): 60 trades, 96.7% WR, +13.8% avg P&L**
**CAUTION (6): 13 trades, 92.3% WR, +2.7% avg P&L**
**NO-GO (<6): 20 trades, 35.0% WR, -3.2% avg P&L**

### Pre-Trade 7-Criteria
**GO (6-7): 45 trades, 95.6% WR, +14.8% avg P&L**
**CAUTION (5): 20 trades, 95.0% WR, +6.4% avg P&L**
**NO-GO (<5): 28 trades, 53.6% WR, +0.3% avg P&L**

### Intensity Score Thresholds
| Threshold | Trades | Win Rate | Avg P&L |
|-----------|--------|----------|---------|
| >=65 | 21 | 95% | +19.2% |
| >=50 | 47 | 96% | +14.8% |
| <50 | 46 | 70% | +2.3% |
| <30 | 20 | 40% | -2.7% |

## Current State
- All 4 files updated and verified working end-to-end for 93-trade dataset
- `bounce_scorer.py` produces correct score distribution (33 A+, 27 A, 13 B, 7 C, 13 F)
- `bounce_exit_targets.py` returns correct frameworks/stats per cap
- `bounce_trader.py` TTS alerts reflect 93-trade stats, intensity spec V4
- `generate_report.py` cheat sheet and all bounce stats updated
- No runtime errors — Pyright warnings are pre-existing type-checking noise
- Chart review tool available at `scripts/bounce_chart_review.py`

## Next Steps
- Consider re-deriving intensity spec weights now that correlations have strengthened
- Study next-day follow-through on GO trades (hold overnight analysis)
- Could run multi-year scan (2020-2025) for even more comprehensive dataset
- Could add `--setup-type` filter to backscanner (weakstock only, strongstock only)
- Analyze which setup type performs better in different market regimes

## Key Files
| File | Role |
|------|------|
| `analyzers/bounce_scorer.py` | V2 bounce scoring (8 historical criteria, 7 pre-trade) — profiles updated for 93 trades |
| `analyzers/bounce_exit_targets.py` | Exit target framework — hit rates and dip risk updated for 93 trades |
| `scanners/bounce_trader.py` | Live bounce monitor — TTS alerts and intensity spec V4 |
| `scripts/generate_report.py` | Daily report — cheat sheet and all bounce stats updated for 93 trades |
| `scripts/bounce_chart_review.py` | Streamlit visual review tool with removal flagging |
| `scripts/compute_bounce_stats.py` | Stats computation script — generates bounce stats JSON |
| `scripts/compute_93_stats.py` | Detailed stats computation for 93-trade verification |
| `data/bounce_data.csv` | **93 historical bounce trades** (was 123, was 54) |
| `data/bounce_scored.csv` | Scored output from bounce_scorer.py |
| `scanners/bounce_backscanner.py` | Historical bulk bounce scanner |
| `data/backscanner_cache/` | Pickle cache of fetched grouped daily data |
