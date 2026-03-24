# Research Lessons

## Session 2026-03-12_18-06
- **The "not significant" labels were misleading.** All three experiments were tagged "not significant" by the statistical guard, but the underlying Spearman correlations were highly significant (p < 0.001). In future sessions, look past the experiment-level label and examine the individual test statistics. The guard may be too conservative for exploratory feature screening.

- **Feature importance before stratification was the right order.** Starting with Spearman correlations across all features gave us a roadmap of what matters before we start slicing the data. Future sessions should always run feature importance first, then stratify on the top features — not the other way around. Hypothesis-free exploration → targeted validation is the right sequence.

- **The top features cluster around a single concept.** For reversals, the top 5 features are all measuring "extension." For bounces, the top features are all measuring "selloff severity." This means the strategies are fundamentally simple — they're mean reversion plays, and the magnitude of the deviation is the dominant predictor. Future research should focus on **secondary factors** (volume, regime, cap) that modulate this primary driver, not on discovering new primary drivers.

- **The regime analysis was underpowered for fine-grained splits.** With N=95 bounces split across 3 SPY daily buckets and 3 SPY trend buckets, we're down to N=7-67 per cell. Interaction effects (regime × feature) will be even worse. For the bounce strategy, we may need to either (a) expand the dataset via backscanning, or (b) limit ourselves to binary splits (favorable/unfavorable) rather than 3-way splits.

- **Cap stratification was not tested this session — and it should be next.** The CLAUDE.md notes that Micro/Small/Medium/Large/ETF have different dynamics, and the existing threshold system is cap-adjusted. But we didn't validate whether cap size actually matters for expectancy in this session. This is a known important variable we skipped. Prioritize it next time.

---


## Session 2026-03-16_20-03
- **Cap stratification changes the story.** The all-cap reversal feature importance screams "extension is everything." But within Medium caps (58% of the dataset), volume metrics are the top predictors. This means the all-cap analysis was being driven partly by cross-cap variance, not just within-cap predictive power. Future sessions should ALWAYS stratify by cap before drawing conclusions about feature importance.

- **Setup type classification is a powerful edge for bounces.** The entry_signal_comparison was the most impactful experiment this session. Knowing that IntradayCapitch_strongstock has 14% WR versus GapFade_weakstock at 96% WR is immediately actionable. The setup type acts as a coarse but effective pre-filter. We should run the equivalent analysis for reversals once we have more typed setups beyond 3DGapFade.

- **Exit optimization deserves more attention than entry optimization.** We've spent 3 sessions refining entry filters and have a well-confirmed set (spy_5day, gap_pct, pct_from_9ema). But the exit analysis reveals 67.5% of available profit is being left behind. A 25% trailing stop alone would boost reversal expectancy from +13.5% to +17.3%. Future sessions should dedicate at least one experiment to exit rules — they compound with entry improvements.

- **Walk-forward "NO_EDGE" verdicts are misleading for high-WR strategies.** All 3 bounce splits were labeled "collapsed," yet the OOS win rates averaged 90.3% and OOS P&L averaged +6.22%. The walk-forward engine's collapse threshold appears miscalibrated for strategies that already have 80%+ baseline win rate. The engine seems to be comparing GO-filtered trades against the full sample — when the filter selects fewer trades OOS (as expected), it's calling that "collapse." We need to either recalibrate the engine or interpret its verdicts more carefully.

- **The "not significant" labels from the statistical guard continue to be too conservative.** The reversal Medium-cap feature importance was labeled "not significant" despite 14 of 29 features having Spearman p-values < 0.05. The entry signal comparison had a pairwise p=0.007 and was correctly labeled significant. The inconsistency suggests the guard is applying a blanket Bonferroni-style correction that's appropriate for filter sweeps but too strict for feature screening. Always check individual test statistics.

---
