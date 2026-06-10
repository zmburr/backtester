# Signal Findings Ledger

Canonical record of what the signal feedback loop (scorecard → analysis) has
settled, so closed questions stop getting relitigated. Updated when an
analysis cycle closes or reopens an item. All rates are **episode-level**
(first qualifying flag per episode) unless noted; "hit" = 1-ATR favorable
move within the D0..D+3 window.

## Closed — do not relitigate without new evidence

| Question | Verdict | Evidence (as of 2026-06-10, 67 episodes / 147 signals) |
|---|---|---|
| Wait for D0 close-direction confirmation before entry? | **Rejected** | Lift is modest (48% vs 37%) and it forfeits 14/28 winners including all 9 same-day payers. |
| Go long first on early reversal signals ("ride the blow-off")? | **Rejected** | Winners' median adverse run before the crack is 0.34 ATR (p75 0.82); only 12/75 squeezed ≥1 ATR. The big long-side runs occur on signals that never work, unidentifiable ex-ante. |
| Entry-delay variants generally | **Stop testing** | The edge levers are the RVOL veto, the reprint trigger, and the 1.5-ATR stop — not entry timing tweaks. |
| Initial stop placement | **1.5 ATR** | 27/28 winners survive a 1.5-ATR stop; 25/39 losers get capped. Sub-0.5-ATR stops shake out winners. |

## Deployed rules

| Rule | Deployed | Effect (episode-level retest) | Forward verification |
|---|---|---|---|
| RVOL veto: reversal GO/CAUTION with `prior_day_rvol < 1.25` → `VETO` | 2026-06-10 (`generate_report.score_pretrade_setup`) | Baseline 28/67 (41.8%) → 29/49 (59.2%); stable in both halves (33→57%, 50→62%). Skips 18 episodes of which only 7 would have hit. | Scorecard tracks the `Vetoed (RVOL)` cohort; analysis loop flags drift toward the GO rate. |
| Reprint trigger surfaced in priority report (print #, prior-print-paid badge) | 2026-06-10 (`priority_report`) | Reprint after an observable paid prior print: 10/12 (83%) by episode; 28/30 (93%) per-signal. Reprints without a paid prior print: 18/40 (45%). | Banner + `print_num`/`prior_print_paid` in signals JSON. |

## Open — tracking, not yet actionable

- **gap_pct as a reversal criterion**: zero pooled separation (winner median
  0.014 vs loser 0.011, n=67). Retest of the proposed replacement (gap point
  → RVOL≥2.0 tier point) scored **54.5% (18/33) — worse than the plain veto's
  59.2%** with less coverage, so the restructure is NOT adopted. Re-evaluate
  at ~100 first-flags.
- **4/5 > 5/5 score inversion**: persists post-veto (4/5: 15/22 = 68% vs 5/5:
  5/12 = 42%), so sub-1.25-RVOL does not explain it. n(5/5) < 20 — keep
  tracking before restructuring the score.
- **Large-cap weakness**: baseline 9/32 (28%) → 11/22 (50%) post-veto. Big
  improvement but still below Medium; re-measure after ~2 weeks of live veto
  before pursuing a breadth/regime gate.
- **Bounce side**: deferred at 7 independent episodes (4/7). Red flag on
  file: the lone bounce GO (CAR, 5/6) was the dataset's worst signal
  (prior_day_range_atr 4.1 — possible mid-crash knife-catch inversion).
  Revisit at n≥20 episodes.

## Methodology notes

- Episodes: consecutive prints for a ticker within 3 trading days share an
  episode (`episode_id` in `data/signal_outcomes.csv`); windows overlap, so
  per-signal rates overstate independence. Statistical claims use first
  flags or episode counts.
- The 2026-06-10 retest (`scripts/retest_signal_rules.py`) validates rules
  on the same sample they were derived from — internal consistency plus a
  first/second-half stability split, not out-of-sample proof. Forward proof
  accrues in the live scorecard cohorts.
