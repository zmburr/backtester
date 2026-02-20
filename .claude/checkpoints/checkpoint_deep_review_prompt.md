# Checkpoint: deep_review_prompt
**Created**: 2026-02-20
**Status**: completed

## Summary
Rewrote the nightly cron job code review prompt for the backtester project. Replaced vague "pick 3-4 files" with an explicit 8-file tiered review list ranked by P&L impact, with per-file review directives and a structured output format including feature gap analysis.

## What Was Done
- Analyzed all Python files in the backtester project by line count and importance
- Identified 8 critical files across 3 tiers (scoring logic, screening logic, data integrity)
- Rewrote the Codex + Claude Code fallback prompt with:
  - Explicit file list with line counts
  - Per-file review directives (what specifically to check in each file)
  - Tiered priority by P&L impact
  - Explicit exclusions to prevent token waste
  - Structured output: (a) bugs, (b) silent failures, (c) feature gaps
- Updated output format to include feature/strategy improvement suggestions with trading-specific examples

## Key Decisions & Findings
- `main_backtester.py` was dropped from the review — it's just an execution harness, not where P&L risk lives
- Live monitors (bounce_trader.py, reversal_trader.py) excluded — they're ~2,750 lines combined and should be a separate review
- backtesting_strategies/ excluded — small (59-166 lines per file) and stable
- Total review scope: ~4,500 lines across 8 files, fits within $5 Claude Code budget (~55-60K tokens for reading + analysis room)
- V2→V3 scoring migrations are flagged as specific review targets (pct_change_3 replaced consecutive_up_days in reversal, vol_expansion replaced in bounce)

## Current State
Final prompt is complete and ready to paste into the cron job configuration. No files in the repo were modified.

## Next Steps
1. Paste the final prompt into the OpenClaw cron job (ID: 97801ee3-12c1-46cb-bedd-6dc1f1b59483)
2. Consider creating a separate cron job for live monitor review (bounce_trader.py + reversal_trader.py)
3. After a few nights of reviews, evaluate whether the feature gap suggestions are actionable or need more specific anchoring

## Key Files
The 8 files in the review, tiered by priority:

### TIER 1 — Scoring Logic
- `analyzers/reversal_scorer.py` (615 lines) — Parabolic short scoring, V3 dual-score system
- `analyzers/reversal_pretrade.py` (489 lines) — Per-setup-type reversal validator (3DGapFade)
- `analyzers/bounce_scorer.py` (1160 lines) — Bounce pre-trade checklist + setup classification
- `analyzers/bounce_exit_targets.py` (422 lines) — ATR-based exit framework

### TIER 2 — Screening Logic
- `scanners/setup_screener.py` (969 lines) — Universe screener (parabolic + bounce)
- `scanners/historical_backscanner.py` (648 lines) — Bulk historical scanner

### TIER 3 — Data Integrity
- `data_queries/polygon_queries.py` (378 lines) — Polygon API wrapper
- `data_collectors/bounce_collector.py` (875 lines) — 50+ metric feature engineering

## Final Prompt
The complete prompt is in the conversation history. Key structure:
- Step 1: Codex exec with full file list + per-file directives
- Step 2: Claude Code fallback (same directives, slightly condensed)
- Step 3: Telegram summary to -1003668845171:18
- Schedule: 30 22 * * 1-5 (10:30 PM MT, weekdays)
- Timeout: 3600s
