# Checkpoint: bounce_scorer
**Created**: 2026-02-19 17:30
**Status**: completed

## Summary
Built a historical bounce backscanner, expanded the training set from 54 to 123 trades across 4 crash periods, then updated all hardcoded statistics across 4 files (bounce_scorer, bounce_exit_targets, bounce_trader, generate_report) to reflect the larger dataset. All updates verified working.

## What Was Done
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
- Created `data/bounce_stats_123.json` — reference stats for all updates
- **Updated `analyzers/bounce_scorer.py`**:
  - Docstring: 54 to 123 trades
  - Weakstock profile: sample_size=28, wr=0.86, avg_pnl=9.1 (was 16, 0.75, 13.3)
  - Strongstock profile: sample_size=40, wr=0.83, avg_pnl=7.4 (was 24, 0.71, 2.7)
  - Updated all reference_medians for both profiles
- **Updated `analyzers/bounce_exit_targets.py`**:
  - Large cap now has own data (n=24, was n=1 using Medium defaults) — 96% hit 0.5x ATR
  - All cap sample sizes: ETF 16, Medium 75, Small 8, Large 24
  - Dip risk dramatically reduced: ETF avg -2.5% (was -14%), Large avg -2.5% (was -14%)
  - Hit rates updated across all caps and tiers
- **Updated `scanners/bounce_trader.py`**:
  - Intensity spec V3: all 7 correlation values updated for 123-trade dataset
  - Drawdown alerts: median -0.4 ATR (was -1.3), p75 -1.1 ATR (was -2.8)
  - Gap fill TTS: 86% half-fill (was 70%), 65% full (was 48%)
  - Setup alert stats updated for both weak/strongstock
  - Time alerts: early low 99% WR (was 100%), retained 55% (was 51%)
- **Updated `scripts/generate_report.py`**:
  - All "54 trades" references to "123 trades"
  - Bounce cheat sheet: ATR targets, gap fill rates, down-day stats, Bollinger, near lows
  - Intensity spec correlations updated to match bounce_trader.py
  - Exit target framework stats per cap
  - Profile stats for both setup types

## Key Decisions & Findings
- **Dip risk was massively overstated** on old 54-trade data: avg -14% / 1.68 ATRs -> actual ~4% / 0.4 ATRs
- **Strongstock performance much better** than old data suggested: 83% WR, +7.4% (was 71%, +2.7%)
- **Correlations weaker with larger sample**: selloff rho -0.337 (was -0.712), suggesting intensity weights may need re-derivation
- **Large cap now standalone**: n=24 gives enough data for own exit framework (96% hit 0.5x ATR)
- **Gap fill rates improved**: 65% full fill (was 49%), 86% half fill (was 78%)
- **Leveraged/inverse ETPs excluded** from training set
- **Default cap filter is `Large,ETF,Medium`**

## Scoring Results (123 trades, 8 criteria)
| Score | Trades | Win Rate | Avg P&L |
|-------|--------|----------|---------|
| 8/8 | 34 | 100% | +17.5% |
| 7/8 | 42 | 88.1% | +8.0% |
| 6/8 | 24 | 79.2% | +1.8% |
| 5/8 | 10 | 60.0% | +0.3% |
| <=4/8 | 13 | 15.4% | -5.2% |

**GO: 76 trades, 93.4% WR, +12.2% avg P&L**
**CAUTION: 24 trades, 79.2% WR, +1.8% avg P&L**
**NO-GO: 23 trades, 34.8% WR, -2.7% avg P&L**

## Current State
- All 4 files updated and verified working end-to-end
- `bounce_scorer.py` loads profiles correctly, produces expected score distribution
- `bounce_exit_targets.py` imports and returns correct frameworks/stats
- `bounce_trader.py` TTS alerts reflect 123-trade stats
- `generate_report.py` cheat sheet and intensity spec updated
- No runtime errors — Pyright warnings are pre-existing type-checking noise

## Next Steps
- Could run multi-year scan (2020-2025) for even more comprehensive dataset
- Study next-day follow-through on GO trades (hold overnight analysis)
- Consider re-deriving intensity spec weights now that correlations have weakened
- Could add `--setup-type` filter to backscanner (weakstock only, strongstock only)
- Analyze which setup type performs better in different market regimes

## Key Files
| File | Role |
|------|------|
| `analyzers/bounce_scorer.py` | V2 bounce scoring (8 historical criteria, 7 pre-trade) — profiles updated |
| `analyzers/bounce_exit_targets.py` | Exit target framework — hit rates and dip risk updated |
| `scanners/bounce_trader.py` | Live bounce monitor — TTS alerts and intensity spec V3 |
| `scripts/generate_report.py` | Daily report — cheat sheet and all bounce stats updated |
| `scripts/compute_bounce_stats.py` | Stats computation script — generates bounce_stats_123.json |
| `data/bounce_stats_123.json` | Reference stats JSON for 123-trade dataset |
| `data/bounce_data.csv` | **123 historical bounce trades** (was 54) |
| `data/bounce_scored.csv` | Scored output from bounce_scorer.py |
| `scanners/bounce_backscanner.py` | Historical bulk bounce scanner |
| `data/backscanner_cache/` | Pickle cache of fetched grouped daily data |
