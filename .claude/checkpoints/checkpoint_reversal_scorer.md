# Checkpoint: reversal_scorer
**Created**: 2026-02-19 (updated)
**Status**: completed

## Summary
Upgraded reversal_scorer.py from V2 to V3: added a pre-trade score (criteria 1-5, no outcome leakage from criterion 6), a continuous ATR-adjusted intensity score (0-100) with cap-stratified percentile ranking, and a longer-term extension metric (pct_from_50mav).

## What Was Done

### V2 (earlier session)
- Replaced non-predictive `consecutive_up_days` (rho=0.086) with `pct_change_3` (rho=+0.546) across all scoring files
- Created analysis scripts: `reversal_predictive_analysis.py`, `reversal_v2_prototype.py`
- Updated `reversal_scorer.py`, `reversal_pretrade.py`, `setup_screener.py`, `historical_backscanner.py`, `generate_report.py`
- Committed as `c6beecd`

### V3 (this session)
- `analyzers/reversal_scorer.py` — major additions:
  - Added `_pctrank()` helper (scipy with numpy fallback, mirrors bounce_trader.py)
  - Added `_REVERSAL_INTENSITY_SPEC` — 6 weighted components: ema9_atr (0.20), mom3_atr (0.20), pct_from_50mav (0.15), prior_day_range_atr (0.15), gap_atr (0.15), rvol_score (0.15)
  - Added `_CAP_GROUPS` dict and `_ref_by_cap_group` — cap-stratified reference sets (Micro/Small vs Medium/Large/ETF)
  - Added module-level Grade-A reference data loading with precomputed ATR-adjusted columns
  - Added `compute_reversal_intensity(metrics, cap)` — standalone function, cap-stratified percentile ranking
  - Added `_pretrade_grade()` and `_pretrade_recommendation()` static methods (5-point scale)
  - Updated `score_setup()` — now returns pretrade_score/grade/recommendation + intensity
  - Updated `score_dataframe()` — adds pretrade_score, pretrade_grade, pretrade_recommendation, intensity columns
  - Updated `print_score_report()` — shows [PRE]/[OUT] tags, pre-trade line, intensity score
  - Rewrote `__main__` block — pretrade distribution, intensity quartiles, Spearman correlations

## Key Decisions & Findings

### V2 Findings
- `consecutive_up_days` NOT predictive (rho=+0.086, p=0.35) — replaced with `pct_change_3` (rho=+0.546)
- Classification gate preserved: `classify_reversal_setup()` still uses `consecutive_up_days >= 2`
- V2 GO: 96% WR, +14.6%

### V3 Findings
- **Binary score destroys signal**: pass/fail 6-criterion score has weak rho=+0.222 vs magnitude. Continuous ATR-adjusted metrics are far better (rho up to +0.812).
- **Criterion 6 is outcome leakage**: reversal_open_close_pct IS the outcome. Pre-trade score uses only criteria 1-5.
- **Pre-trade grade mapping (5-point)**: 5/5=A+ GO, 4/5=A GO, 3/5=B CAUTION, 0-2=C/F NO-GO
- **Cap-stratified ranking chosen over global**: Comparing within Micro/Small or Medium/Large/ETF peer groups. Lifts MSTR 11/21/2024 from intensity 39→50.
- **pct_from_50mav added**: Raw % above 50MA captures longer-term extension. rho=+0.409 with P&L. MSTR was 138% above 50MA = 76th pctile among Medium+ caps.
- **atr_distance_from_50mav rejected**: ATR-adjusted version has negative/insignificant rho=-0.110. Raw % works better.
- **rvol_score kept**: Weak rho=+0.187 but still useful as binary GO/NOGO filter.

### Key Results
- Pre-Trade GO (41 trades): 97.6% WR, +14.6% avg P&L
- Pre-Trade !GO (11 trades): 63.6% WR, +9.9% avg P&L
- Intensity quartiles (cap-stratified): Q1 +8.9% → Q2 +7.6% → Q3 +20.9% → Q4 +21.7%
- Intensity vs P&L: rho=+0.394 (p=0.011)

### Trade Spot-Checks
| Trade | Pre-Trade | Intensity | P&L |
|-------|-----------|-----------|-----|
| MSTR 11/21/2024 | 5/5 A+ GO | 50 | +25.8% |
| GLD 10/17/2025 | 5/5 A+ GO | 38 | +2.1% |
| GLD 1/29/2026 | 5/5 A+ GO | 55 | +2.7% |
| MSTR 10/29/2024 | 1/5 F NO-GO | N/A | +2.3% |
| MSTR 1/11/2024 | 3/5 B CAUTION | N/A | +10.5% |

## Current State
- `analyzers/reversal_scorer.py` fully updated to V3 and tested, runs clean
- `data/reversal_scored.csv` regenerated with new columns
- No changes to downstream consumers (setup_screener.py, historical_backscanner.py, reversal_pretrade.py)
- MSTR 11/21 intensity at 50 (up from 39 in V3a) — prior_day_range_atr at 14th pctile is biggest drag

## Next Steps
- Consider whether intensity should feed into `setup_screener.py` output for live screening
- Consider whether a reversal trade monitor (like bounce_trader.py) should display intensity in real-time
- Could explore multi-day range expansion (not just prior day) to improve the range signal
- May want to integrate intensity into `scripts/generate_report.py` daily reports

## Key Files
- `analyzers/reversal_scorer.py` — Main file (V3: pretrade score + cap-stratified intensity)
- `analyzers/reversal_pretrade.py` — Per-setup-type validator (3DGapFade), independent from scorer
- `analyzers/reversal_predictive_analysis.py` — Spearman correlation analysis (V2 research)
- `scanners/setup_screener.py` — Universe screener, downstream consumer (not yet updated for intensity)
- `scanners/historical_backscanner.py` — Bulk historical scanner (not affected)
- `scripts/generate_report.py` — Daily report (not yet updated for intensity)
- `data/reversal_data.csv` — 120 historical reversal trades (52 Grade A)
- `data/reversal_scored.csv` — Output with pretrade + intensity columns
