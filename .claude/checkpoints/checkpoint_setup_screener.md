# Checkpoint: setup_screener
**Created**: 2026-02-15
**Status**: in_progress

## Summary
Built per-setup-type reversal scanners (starting with 3DGapFade) and a historical backscanner that finds matching setups across all tickers and dates using Polygon's grouped daily endpoint. Replaces the generic one-size-fits-all ReversalScorer thresholds with data-driven per-cap thresholds derived from 33 historical 3DGapFade trades.

## What Was Done
- **Created `analyzers/reversal_pretrade.py`** — Per-setup-type reversal pre-trade validator mirroring `BouncePretrade` pattern. Contains `ReversalSetupProfile` dataclass, `classify_reversal_setup()` auto-detection, `ReversalPretrade` class with `validate()` and `print_checklist()`. Reuses `ChecklistItem`/`ChecklistResult` from `bounce_scorer.py`.
- **Modified `scanners/setup_screener.py`** — Integrated `ReversalPretrade`. `_score_parabolic_short()` now tries typed classification first (3DGapFade), falls back to generic `ReversalScorer`. Added `parabolic_setup_type` field to `ScreenResult`. Typed setups require score >= 4 (GO), generic keeps >= 3. Output table shows detected setup type.
- **Created `scanners/historical_backscanner.py`** — Historical backscanner using `get_grouped_daily_aggs()` (1 API call/date for ALL tickers). Pre-builds ticker-indexed lookup for O(1) per-ticker access. Quick pre-filters (price > $5, avg vol > 500K, consecutive up days, gap direction) before full metric computation. Caches to pickle for resume. CLI via `python -m scanners.historical_backscanner`.
- **Updated `CLAUDE.md`** — Added descriptions for both new files, CLI commands, updated project structure tree.
- **Generated `data/backscanner_2021_A_plus.csv`** — 457 A+ (5/5) 3DGapFade setups found in 2021.
- **Generated `data/backscanner_results_3DGapFade_2021-01-01_2021-12-31.csv`** — Full 2021 scan results (18,185 matches total).

## Key Decisions & Findings

### 3DGapFade Classification Gate
All three must be true:
- `consecutive_up_days >= 2` (2+ prior euphoric up days)
- `gap_pct > 0` (gaps up on the fade day)
- `pct_from_9ema > 0.04` (4%+ above 9EMA)

### Per-Cap Scoring Thresholds (3DGapFade)
| Criterion | ETF | Large | Medium | Small | Micro |
|-----------|-----|-------|--------|-------|-------|
| pct_from_9ema | 0.05 | 0.08 | 0.15 | 1.00 | 1.20 |
| prior_day_range_atr | 1.40 | 0.93 | 1.00 | 1.95 | 3.30 |
| rvol_score | 1.80 | 1.35 | 1.50 | 6.00 | 1.60 |
| consecutive_up_days | 2 | 2 | 2 | 3 | 2 |
| gap_pct | 0.00 | 0.00 | 0.00 | 0.50 | 0.20 |

Scoring: 5/5 = A+ GO, 4/5 = A GO, 3/5 = B CAUTION, <3 = F NO-GO

### Historical Validation Results
- **CSV validation**: 27/33 historical 3DGapFade trades caught (82%). All 6 misses fail the classification gate (not scoring). CAUGHT: 89% WR, +17.4% avg P&L. MISSED: 83% WR, +2.5% avg P&L (weaker setups correctly filtered).
- **Backscanner smoke test**: 2024-11-21 found MSTR at 5/5 A+ in 10.4 seconds across 12,397 tickers.
- **2021 full year**: 457 A+ setups found. Caught GME (9 days), AMC (9 days), DWAC, CLOV, SPRT, LCID, IONQ, RBLX, NVDA, etc. Jan/Feb were hottest months (meme stock era).

### Performance Optimization
Original `_build_ticker_history` did per-ticker DataFrame filtering across all dates — O(tickers * dates). Fixed by pre-building a ticker-indexed dict from concatenated data (one-time O(total_rows)), then O(1) lookup per ticker. Reduced single-date scan from "hung indefinitely" to ~10 seconds for 12K tickers.

### Missed Trades Analysis
6 trades missed by classification gate:
- 5 had `consecutive_up_days < 2` (BBBY, AVGO, KWEB, VXX, MSTR 10/29)
- 1 had `gap_pct <= 0` (CRCL, which was also the only loser at -10.5%)
- These missed trades averaged only +2.5% PnL vs +17.4% for caught trades

## Current State
- All code is working and tested. Not yet committed to git.
- Cached data exists for 2021 scan (`data/backscanner_cache/grouped_daily_2020-02-26_2021-12-31.pkl`) and 2024-11-21 smoke test.
- User has `data/backscanner_2021_A_plus.csv` to review.

## Next Steps
1. **User manually labels `data/backscanner_2021_A_plus.csv`** — Mark each row as "should have fired" or "should not have fired" (add a column like `valid` = Y/N or `label` = true_setup / false_positive)
2. **Refine criteria based on labeled data** — Compare true setups vs false positives across all 5 criteria dimensions. Identify which thresholds or classification rules need tightening/loosening to maximize true positives and minimize false positives.
3. After refinement: add next-day performance tracking to backscanner output (fetch day-after open/close to measure actual reversal P&L)
4. Add more setup types beyond 3DGapFade (e.g., 2DGapFade, 2DBreakoutIB, ConsolidationBreakdown)
5. Run backscanner on other years (2020, 2022-2025) for broader validation
6. Consider committing changes to git

## Key Files
| File | Description |
|------|-------------|
| `analyzers/reversal_pretrade.py` | NEW — Per-setup reversal pre-trade validator (3DGapFade profiles + classification) |
| `scanners/setup_screener.py` | MODIFIED — Integrated ReversalPretrade, typed setup detection |
| `scanners/historical_backscanner.py` | NEW — Bulk historical scanner using grouped daily Polygon endpoint |
| `data/reversal_data.csv` | Source of truth — 120 reversal trades (33 are 3DGapFade) |
| `analyzers/reversal_scorer.py` | EXISTING — Generic reversal scorer (now used as fallback only) |
| `analyzers/bounce_scorer.py` | EXISTING — Source of ChecklistItem/ChecklistResult dataclasses |
| `data/backscanner_2021_A_plus.csv` | OUTPUT — 457 A+ 3DGapFade setups from 2021 |
| `data/backscanner_results_3DGapFade_2021-01-01_2021-12-31.csv` | OUTPUT — Full 2021 scan (18,185 matches) |
| `data/backscanner_cache/` | Cached Polygon grouped daily data (pickle files) |
| `CLAUDE.md` | MODIFIED — Updated with new file descriptions and CLI commands |
