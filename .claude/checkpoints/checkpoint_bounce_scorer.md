# Checkpoint: bounce_scorer
**Created**: 2026-02-19 17:30
**Updated**: 2026-02-20 16:00
**Status**: completed

## Summary
Built a historical bounce backscanner, expanded the training set from 54 to 123 trades across 4 crash periods, visually reviewed all new trades (removed 30, down to 92), then isolated IntradayCapitch trades from all reference data, made all comparisons setup-specific (weakstock vs strongstock), and re-derived intensity spec weights (V5). All 4 files use clean GapFade-only stats (83 trades).

## What Was Done

### Phase 1: Expansion (Feb 19)
- Created `scanners/bounce_backscanner.py` (~500 lines) — historical bulk bounce scanner
- Expanded `data/bounce_data.csv` from 54 to 123 trades across 4 crash periods
- Created `scripts/compute_bounce_stats.py` — generates comprehensive stats JSON

### Phase 2: Visual Review & Cleanup (Feb 20 AM)
- Created `scripts/bounce_chart_review.py` — Streamlit visual review tool
- Visually reviewed all 69 new trades, flagged 30 for removal → 92 trades
- Updated all 4 stat files

### Phase 3: IntradayCapitch Isolation & Setup-Specific Data (Feb 20 PM)
- **Root problem**: 9 IntradayCapitch trades (11% WR, -10.2% avg) were contaminating all GapFade stats
  - `classify_from_setup_column()` lumped IntradayCapitch into GapFade_strongstock
  - All reference DataFrames, intensity scores, percentile comparisons used aggregate data
  - IntradayCapitch was dragging down CAUTION/NO-GO stats and creating false SPY correlation

- **`analyzers/bounce_scorer.py`**:
  - `classify_from_setup_column()` now returns `'IntradayCapitch'` as distinct category
  - Profile stats updated: Weakstock 36 A-grade/92% WR/+12.9%, Strongstock 31 A-grade/97% WR/+10.9%

- **`scripts/generate_report.py`**:
  - `_bounce_df_all` filters out IntradayCapitch before building BOUNCE_DF_WEAK/BOUNCE_DF_STRONG
  - Intensity score now uses setup-specific ref_df (BOUNCE_DF_WEAK or BOUNCE_DF_STRONG)
  - Bounce ticker sorting uses setup-specific ref_df via classify_stock()
  - BOUNCE_SCORE_STATISTICS updated: CAUTION 19/100%/+9.3%, NO-GO 20/70%/+2.5%
  - All cheat sheet stats recomputed from 83 GapFade trades
  - Correlations updated: pct_change_3 now #1 (rho=-0.539), SPY dropped to insignificant (rho=0.09)

- **`scanners/bounce_trader.py`**:
  - `_bounce_df_all` filters out IntradayCapitch; split into `_bounce_df_weak` and `_bounce_df_strong`
  - Intensity score uses setup-specific ref_df based on self.setup_type
  - TTS alerts updated: weakstock +21%/+11%, strongstock +12%/+8%, late low 50% WR, overnight 98%/+14.2%

- **`analyzers/bounce_exit_targets.py`**:
  - All hit rates and dip risk updated: ETF n=15, Medium n=45, Large n=16 (100% hit at 0.5ATR!)

### Phase 4: Intensity Spec V5 (Feb 20 PM)
- Re-derived weights from within-setup Spearman correlations
- Key findings:
  - **Multicollinearity severe**: pct_off_30d_high and pct_off_52wk_high rho=0.885
  - **Within-setup predictors differ**: pct_change_15 #1 for weakstock (-0.694) but insignificant for strongstock
  - **consecutive_down_days not significant** for either setup within-setup
  - **SPY correlation was false signal** driven by IntradayCapitch (0.244 → 0.09)

- V5 weight changes:
  - pct_change_3: 0.20 → **0.30** (#1 predictor both setups)
  - pct_change_15: 0.05 → **0.20** (#1 weakstock rho=-0.694)
  - selloff_total_pct: 0.25 → **0.15** (multicollinear, reduced)
  - gap_pct: 0.15 → 0.15 (unchanged, #2 strongstock)
  - pct_off_30d_high: 0.15 → 0.15 (unchanged)
  - pct_off_52wk_high: 0.10 → **0.05** (redundant with 30d, rho=0.885)
  - consecutive_down_days: 0.10 → **removed** (not significant within-setup)

- V5 performance vs V4:
  - PnL gap: +10.5% → **+11.2%**
  - Weak rho: 0.665 → **0.690**
  - Strong rho: 0.456 → **0.479**

## Key Decisions & Findings
- **IntradayCapitch excluded from all reference data** — 9 trades with 11% WR were polluting stats
- **Setup-specific intensity scoring**: each trade compared against its own setup type
- **V5 intensity weights**: re-derived from within-setup correlations, dropped non-predictive metrics
- **SPY correlation was FALSE SIGNAL**: driven by IntradayCapitch, dropped from rho=0.244 to 0.09
- **Gap fill much better than previously reported**: 91% half fill, 70% full fill, 54% close above gap

## Scoring Results (83 GapFade trades)

### Pre-Trade 7-Criteria
**GO (6-7): 45 trades, 95.6% WR, +14.8% avg P&L**
**CAUTION (5): 19 trades, 100.0% WR, +9.3% avg P&L**
**NO-GO (<5): 20 trades, 70.0% WR, +2.5% avg P&L**

### Intensity Score V5 (setup-specific ref)
| Threshold | Trades | Win Rate | Avg P&L |
|-----------|--------|----------|---------|
| >=65 | 23 | 96% | +19.4% |
| >=50 | 38 | 97% | +16.6% |
| <50 | 45 | 84% | +5.4% |
| <30 | 15 | 60% | -0.1% |

## Current State
- All 4 files updated and verified end-to-end for 83 GapFade trades
- IntradayCapitch fully excluded from all reference DataFrames
- Setup-specific ref_df used for intensity scores in both generate_report and bounce_trader
- V5 intensity weights deployed (6 metrics, consecutive_down_days removed)
- No runtime errors

## Next Steps
- Study next-day follow-through on GO trades (hold overnight analysis)
- Could run multi-year scan (2020-2025) for even more comprehensive dataset
- Could add `--setup-type` filter to backscanner (weakstock only, strongstock only)
- Analyze which setup type performs better in different market regimes

## Key Files
| File | Role |
|------|------|
| `analyzers/bounce_scorer.py` | V2 bounce scoring — profiles updated, IntradayCapitch classified separately |
| `analyzers/bounce_exit_targets.py` | Exit target framework — hit rates updated for 83 GapFade trades |
| `scanners/bounce_trader.py` | Live bounce monitor — setup-specific intensity V5, TTS alerts updated |
| `scripts/generate_report.py` | Daily report — setup-specific intensity V5 + percentiles, all cheat sheet stats updated |
| `scripts/bounce_chart_review.py` | Streamlit visual review tool with removal flagging |
| `scripts/compute_93_stats.py` | Stats computation script (filters IntradayCapitch) |
| `data/bounce_data.csv` | **92 trades total** (83 GapFade + 9 IntradayCapitch, IC excluded from stats) |
| `scanners/bounce_backscanner.py` | Historical bulk bounce scanner |
