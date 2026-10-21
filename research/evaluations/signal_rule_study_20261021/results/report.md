# Signal Scorecard: completed rule and exit study

October 21, 2026 · active Mac mini production snapshot after the historical repairs.

**Decision:** retain the current thresholds and simulated exits. The study found real profit givebacks, but selling automatically at +1R hurt the measured return. Break-even protection is a small, uncertain candidate; no individual signal rule has enough robust evidence here to remove it. Exact decision logging and the quiet-day control-record fix are installed on the Mac mini.

## What was evaluated

- 2,783 existing scorecard rows, including 2,115 watchlist bounce rows; Method breadth records were excluded.
- 1,876 bounce rows have all four daily bars. 1,331 produced a strict simulated entry; 84 produced no entry (return zero); 461 lack a complete minute session and were excluded from execution comparisons.
- The most defensible rule sample contains 380 complete records on 40 dates, with 296 usable execution outcomes. It requires a matching source, pre-open timing, no conflicting surfaced/control duplicate, and a date after the March threshold revision.
- Crucially, genuinely recorded pre-open controls with complete horizons exist on only **October 12–15**. The older 1,353 control observations were reconstructed with unknown observation time. They are retained in the broader sensitivity, not silently treated as contemporaneous controls.
- Archived profile identity is absent. A rule is classified pass/fail only when both frozen strong-stock and weak-stock profiles agree. Rounded values at the threshold stay ambiguous. A separate score-compatible reconstruction is explicitly conditional, not an assertion of the historical profile.

## Individual bounce rules

Differences below are **pass minus fail**, matched within date and market cap, then equally weighted across dates. Positive means higher simulated net return for passers. These are associations; the rows differ on other rules too.

| Rule | Matched dates | Net difference | 95% interval |
|---|---:|---:|---|
| Deep selloff | 4 | +0.81R | Too few dates |
| Discount from 30-day high | 1 | +2.66R | Too few dates |
| Gap down | 9 | +0.31R | Too few dates |
| Prior-day range expansion | 10 | +0.21R | [-0.34, +0.75] |
| Three-day decline | 5 | -0.57R | Too few dates |
| Discount from 52-week high | 4 | +0.18R | Too few dates |

Range expansion has the most usable comparison: +0.21R over 10 dates, but its interval spans zero. The broader history gives +0.18R over 30 dates, also spanning zero. The primary estimate remains positive under 15bp-per-side costs (+0.23R) and a 0.5-ATR risk floor (+0.27R); neither establishes a benefit.
The three-day decline changes sign between the primary sample (−0.57R) and broader history (+0.58R). The 30-day discount estimate comes from one date. Treating either as grounds for a threshold change would be overconfident.
Holding the other five rules at exactly three points is the closest available comparison to whether one rule should decide admission. Even after inferring score-compatible profiles, these tests have **zero or one matched date per rule**. Four-session non-overlapping observations also leave only zero to five matched dates. Exact rule removal is not identified by this archive.

## Favorable moves versus captured return

Among 309 surfaced simulated attempts, 126 reached at least +1R on a completed two-minute close before the exit interval. **38 (30.2%) later lost money**: 29 hit the initial stop and 9 closed red at the scheduled session close. This is a genuine giveback measure: it does not credit a high reached after the trade had already stopped out.
For first-signal attempts recorded before the open, 11 of 42 attempts that reached a +1R close later lost money. The stored legacy MFE statistic is looser: 72 of 167 surfaced rows with at least 1R of stored opportunity had negative stored exit R. Those quantities have different entry and timing assumptions and should not be conflated.

Three fixed variants were tested on the **same entries**, without optimizing their thresholds. The preferred exit comparison covers 63 first signals on 30 dates, including zero-return no-entry signals. A signal can have up to two fixed baseline entry attempts. The baseline averaged +0.48R with equal date weighting.

| Exit variant | Average return | Change versus baseline | 95% interval for change |
|---|---:|---:|---|
| Exit all after a +1R close | +0.05R | -0.43R | [-1.43, +0.30] |
| Exit half after a +1R close | +0.27R | -0.21R | [-0.71, +0.15] |
| Move stop to break-even after a +1R close | +0.54R | +0.06R | [-0.12, +0.21] |

Early profit-taking cut off too much upside. Its estimated change remained negative after leaving out any single date (all-out: −0.52R to −0.09R). The break-even variant improved 11 signal outcomes and worsened seven; its +0.06R estimate is small and uncertain. It changes to −0.06R when later repeat signals are included. It is a candidate for a future predeclared shadow comparison, not a demonstrated upgrade.
Entries remain fixed even after a candidate exit would have ended a profitable campaign sooner. This isolates exit management; it is not a full replacement-strategy backtest. Partial-sale fees are proportional, with no additional minimum ticket charge modeled.

Examples of a completed-close gain later given back:

| Signal | Highest completed-close gain before exit | Baseline result | Exit reason |
|---|---:|---:|---|
| NBIS, 2026-08-10 | +4.33R | -0.62R | Session close |
| WDC, 2026-07-29 | +3.54R | -0.45R | Session close |
| RXT, 2026-07-14 | +3.32R | -0.34R | Session close |

## Reversal signals

The archive has 200 complete surfaced reversals and 97 first flags, but **no rejected-reversal control rows**. Only within-surfaced, current-rule associations can be examined; there is no comparable reversal execution replay in this study.
The gap-up rule is the main warning sign: passers had a 20.5-percentage-point lower four-session opportunity rate in 11 matched first-flag dates. Its ordinary interval is −38.6 to −2.3 points, but the multiple-comparison interval reaches zero. Historical rule changes, selection into the surfaced cohort and the small sample prevent a removal recommendation. Other reversal-rule comparisons have only two to nine matched dates. Full tables are in reversal_results.json.

## Changes completed

- Each new priority-report decision now records the actual scoring profile, exact input values, per-rule threshold/operator/pass state, missing-input state, reversal gate outcomes, timestamp with UTC offset, source hashes and a decision ID. The display setup matcher is kept distinct from the scoring profile.
- That evidence travels through the existing surfaced/control JSON files into new scorecard CSV fields. Historical records remain unknown; they were not assigned invented profile identities.
- A quiet report now saves its rejected candidates before returning. All eight existing empty priority files lacked matching control files. The historical omission remains visible; the fix prevents future omissions.
- Installed and verified on the active Mac checkout. Trading thresholds, exits and notification rules were unchanged. Production outcomes and analysis state retained their original hashes during validation.

## Validation and limits

51 local targeted tests pass, including four exit-simulation cases; 13 additional priority/scorecard notification tests pass against staged Mac code. Production-Python integration checks passed both before and after installation, including round-tripping all 2,783 old rows without value changes. Primary rule differences were independently recomputed using explicit date/cap loops.
Reviewed 20 minute-bar charts: the five largest surfaced attempt returns, 12 fixed-seed random ticker-dates and three givebacks. The large winners are sustained moves rather than isolated high prints. Every reviewed trigger met the configured liquidity floors; trade-count fields were absent. The minimum reviewed trigger turnover was $766,591 and prior session turnover $3.62m. This inspection does not establish executable broker fills.
The strict protocol uses completed two-minute signals, next-bar openings, gap-aware stops, a $20k trigger/$1m session turnover floor, and 5bp costs per side. It requires every regular-session minute. Returns are hypothetical R units, not broker P&L. Tick data, spread/queue effects, historical corporate-action availability and fixed ticket charges are not established.
Intervals use a fixed-seed five-observed-date block bootstrap. No interval is reported below 10 matched dates; the JSON also includes multiple-comparison intervals. Everything here is exploratory development evidence, not untouched out-of-sample confirmation. Naive historical timestamps within two hours of the open were conservatively treated as timezone-ambiguous (the Mac currently reports EDT).

## Reproduce and inspect

- `python -m research.scoring_timeline.signal_rule_study --reuse` verifies saved source/bar hashes and recomputes rule tables.
- `python -m research.scoring_timeline.signal_exit_study` recomputes the fixed-entry exit tests.
- `python -m research.scoring_timeline.signal_reversal_rule_study` recomputes the reversal supplement.
- `python -m research.scoring_timeline.write_signal_rule_report` renders this report.
- [Exit comparison](exit_comparison.png) · [20-chart review](chart_review.png) · [Signal rows](audited_rows.csv) · [Exit attempts](exit_attempts.csv) · [Results](results.json)
- `snapshot/` contains the preserved production input. `replay_manifest.json` identifies the existing normalized minute caches reused by this study; those caches remain in their prior research location.
- Mac installation backup: `/Users/zacharyburr/.openclaw/workspace/backtester/research/evaluations/signal_rule_study_20261021/backup`.
