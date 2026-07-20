# Reversal Entry Study — when does the tape say "short now"?

Mirror of `bounce_entry_study` for the short side (2026-07-20). The reversal
GO gate barely separates direction OOS (+2-4pp over ~53% base) — the edge
lives in management and timing. The bounce side answered its timing question
with a validated 2-min break protocol (+3.2R/day); this study asks the same
question for parabolic shorts.

## Question

Which mechanical trigger, fired live off 2-min bars, best times the short
entry on a reversal day — and what does the trigger+stop system actually
earn per day in R?

## Samples

* **Curated** — every row of `reversal_data.csv` (~121 days, winners-only),
  tagged by setup. Primary slice = the gap-fade family (3DGapFade, 2DGapFade,
  GapDownTrendBreak) the management research covers.
* **Controls** — N=150 random `setup_type == "3DGapFade"` days from the
  UNCONDITIONED reversal universe that are NOT in the curated book. This is
  what the bounce study deferred to v2: the honest expectancy of the trigger
  on days a machine (not hindsight) flagged. Seeded sample for reproducibility.

## Detectors (close-based, 2-min bars aggregated from 1-min Polygon)

* `pbb_down`    — 2-min close below the prior 2-min bar's low (mirror of the
  bounce winner).
* `vwap_loss`   — first 2-min close below running VWAP after being above.
* `open_fail`   — 2-min close below today's open after a prior close above it.
* `pm_low_break`— 2-min close below the premarket low.
* `conf2of3`    — 2 of {pbb_down, vwap_loss, open_fail} fired within 10 min.

## Simulation

Enter SHORT at the fire bar's close. Stop = running HOD at entry
(+ buffer x ATR sweep). Stop-out -> re-arm on the next fire (max-attempts
sweep). Exit all remaining at 15:55 close (hold-to-close won the cover-rule
research). R per attempt = (entry - exit) / max(risk, 0.25 ATR) — the
0.25-ATR risk floor guards the tight-stop R inflation the bounce study hit.

Sweep: 5 detectors x entry-window end {10:30, 11:00, 15:30} x stop buffer
{0.0, 0.15 ATR} x max attempts {1, 2, 3} = 90 configs, run on curated AND
control cohorts.

Matrix-overlay check on the winner: exit at the 12:30 tripwire (25% retrace)
when it fires instead of the close — connects this study to the live
MgmtMatrixRules playbook.

## Outputs

`reports/reversal_entry_study/configs_ranked.csv` (curated + control columns
side by side), `top_config_days.csv` (per-day detail for the winner),
console summary. Bars cached under `data/reversal_entry_study/`.

## Port gate (write-down before building the live rule)

The winning detector ports into a short-side `entry_bar_rules` twin in the
morning watcher ONLY if: positive median day R on the curated gap-fade slice
AND positive mean day R on the CONTROL cohort (the honest test) AND stop
rate < 50%. Port with the same armed -> stop-out -> re-arm -> max-2 lifecycle
as the bounce entry rules.
