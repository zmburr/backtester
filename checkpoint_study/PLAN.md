# Checkpoint study — "best action at 6 / 17 / 30 min, in R"

## Question
At 6 / 17 / 30 minutes into a news trade that the live exit rule still holds, does any observable
state say a different action beats holding? The actions are: **exit now**, **tighten to a 1-min
trail**, and **trim half** (which is exactly ½·exit-now). The benchmark is holding on the 2-min
quick trail.

## Why
orderPipe's spoken trend verdict (strong / trending / weak) had **no forward edge**. On a replay of
419 trades (2026-09-29), trending minus weak at +11 min was +0.13 minute-ranges, 95% CI
[-0.30, +0.58]. Yet it was spoken, and adherence treated "weak" as a license to exit. The verdict was
silenced on orderPipe `feat/checkpoint-silence`. This study decides whether anything replaces it.
**The expected result is a kill.**

## Nature
Descriptive, pre-registered, no ML. One train/test split; bucketed conditional means; bootstrap
confidence intervals clustered by date; Holm correction.

## Where
| What | Location |
|---|---|
| Code | `backtester/checkpoint_study/` |
| Shared live code (loaded by path; `ORDERPIPE_ROOT`) | `orderPipe/calculators/checkpoint_features.py` and `calculators/print_filter.py`, `calculators/ref_price_calculator.py` |
| Data | `backtester/data/checkpoint_study/`: `bars5s/`, `panels/`, `report/` |
| Run | with `orderPipe\venv\Scripts\python` (it has sheldatagateway and pymongo) |

## Population (`population.py`)
- **Source:** ExitMonitor `trade_data.csv`, rows tagged `news` and not `[OPT]`.
  - First entry per (symbol, date, side).
  - Anchor between 09:30 and 15:30 ET.
  - Result: 1,781 episodes; 1,225 train / 556 test.
- **Anchor** is when live monitoring would start, i.e. trac_trader's detection time:
  - Start with real seconds → `Start + 11 s`
  - Minute-precision Start → `Start + 30 s + 11 s`
  - The 11 s is the median of Mongo `headline_time` − Start: n=23, IQR 6.5–24.5 s.
- **Ref:** recomputed with the live logic (`ref_price_calculator.get_reference_info_from_bars` on 30s
  bars over the 40 min before the anchor, clamped at 9:30), plus the replay/trac_trader wrong-side
  guard.
- **R per share** = `max(|entry − ref|, 0.15%·entry, 0.5·median(15 pre-entry 1-min ranges))`, via
  `checkpoint_features.r_unit`.
- **Drops** (reported in the funnel): no bars; entry outside the day's range ±2% (split-adjusted or
  bad data).

## Exit rule (`rule_sim.py`)
- Mirrors orderPipe's TradeManager: ref stop for the first 2 min, then `2_min_quick`. The level is
  the latest **completed** 2-min bar's low/high (label strictly before the 5s close-time), bad-print
  filtered, not a ratchet. Breach = 5s close strictly through the level.
- **Parity:** `parity.py` ran the real TradeManager (`process_snapshot`) on 30 random cached trades.
  First breach matched 30/30 to the second. It also matched the live ORCL 9/29 alert, 10:29:30.
- **Survivor at k:** no breach before the checkpoint bar. The trader's actual exit is never an input.
- **Fill:** close of the first 5s bar at or after trigger + 10 s.
- **Horizon:** 120 min, capped at 16:00. An open position at the horizon exits at the last bar,
  flagged `capped`.

## Features at k (`checkpoint_features.extract_features`, bars (anchor, t_k] only)
| Feature | Definition |
|---|---|
| `open_R` | side·(px_k − entry)/R |
| `giveback_R` | MFE (5s closes) − open_R |
| `stall_min` | minutes since the last NEW favourable extreme (a retest doesn't reset) |
| `cushion_R` | side·(px_k − trail level)/R |
| `vol_decay` | volume of the last 2 min ÷ volume of the first 2 min after the anchor |

## Frozen protocol
Committed before any forward outcome is inspected. `analyze.py` stamps the git hash.

1. **Split.** Train: date ≤ 2025-06-30. Test: after.
2. **Eligible checkpoints.** k is analysed only if train survivors ≥ 150 **and** test survivors ≥ 90.
   Decided from counts alone.
3. **Outcome.** `edge_x = fwd_R(x) − fwd_R(hold)`, winsorized at ±5R, for x ∈ {exit_now, trail_1m}
   (+ `switch`, amendment 1).
   One-sided: we only look for actions that BEAT holding.
4. **Cells.**
   - Per eligible k, per feature: terciles from **train** quantiles (1/3, 2/3). Coinciding edges
     merge into fewer buckets.
   - Plus one unconditional cell per k.
   - Each cell is tested for both policies.
5. **Statistics.** Mean edge; 95% percentile bootstrap CI, resampling **dates**, B = 10,000,
   seed 20260929; one-sided p = share of bootstrap means ≤ 0.
6. **Stage A (train selects).** A cell is a candidate if all hold:
   - n_train ≥ 50
   - mean ≥ +0.10R
   - CI lower bound > 0

   Keep the top 6 by CI lower bound. **No candidates → KILL; test outcomes are never computed.**
7. **Stage B (test confirms).** A candidate passes if all hold:
   - n_test ≥ 30
   - mean ≥ +0.10R
   - CI lower bound > 0
   - Holm-adjusted p < 0.05 across the Stage A candidates
8. **Robustness (test, all required).**
   - median > 0 or P(edge > 0) > 50%
   - point estimate > 0 in every variant panel (floor 0.10% / 0.25%, ATR ×0 / ×1, anchor ±30 s,
     print filter off, logged ref, latency 30 s, horizon 60 / 240, 2_min_close trail), each re-bucketed
     by its own train edges
   - positive in ≥ 75% of the calendar quarters in test (only quarters with ≥ 5 cell rows count)
   - mean account-R edge > 0 (move × Max Size ÷ $3k)
9. **Placebo.** 20 runs of the whole protocol under a true null: feature values permuted within k,
   **and** each edge demeaned within (k, split). Shuffling alone isn't a null when the unconditional
   edge is real, because every random bucket inherits it (found in the synthetic mechanics check).
   If ≥ 2 of 20 runs produce any passing cell, the pipeline is flagged and nothing ships.
10. **Kill.** If no cell passes 7 and 8, the checkpoint stays orderPipe's neutral timer (grey marker,
    no voice, verdict record-only) **permanently**. Re-run the frozen protocol only once the test
    population has grown ≥ 50%. New features mean a new, separately registered study.
11. **Ship (only on pass).** `checkpoint_policy.json` holds the passing cells with edge, CI, n and
    text. Live goes through `trader/checkpoint_policy.py` in shadow mode (speech off) for 4 weeks
    or 40 eligible checkpoints before `OP_CHECKPOINT_SPEAK=1`.

## Amendment 1 — two co-primary exit rules (2026-09-29, before any outcome was inspected)
**Trigger:** the trader uses the 2-min **delayed** rule (`2_min_close`) as well as quick, and quick
kills most trades early. Survivor counts only, on the 214 cached episodes:

| Rule | 6 min | 17 min | 30 min |
|---|---|---|---|
| quick | 50% | 13% | 1% |
| delayed | 71% | 31% | 8% |

**Changes** (these supersede the matching points of the frozen protocol):
- **Rules.** `quick` (panel `primary`) and `delayed` (panel `delayed`) each get the full protocol on
  their own panel: eligibility (§2), cells (§4), Stage A (§6) and robustness (§8).
- **New policy.** `switch` means using the OTHER 2-min rule from t_k on (the exit chart's
  Quick/Delayed toggle). Tested policies are now exit_now, trail_1m and switch.
- **Selection.** Stage A keeps ≤ 6 candidates **per rule**. **One Holm correction spans the
  candidates of both rules** (§7).
- **Variants.** 11 sensitivity knobs per rule (`d_` prefix for delayed). The `trail_close` variant is
  dropped, because delayed is now a primary.
- **Placebo.** The null is applied to both rule panels together (§9).
- **Live.** The Phase 3 table is keyed by rule; live uses the table matching `trade.profit_strategy`.

**Found while doing it (a live bug).** On the delayed rule, an entry exactly on an even-minute
boundary means that at 2:00 elapsed the 5s tick lands before that minute's 2-min bar. The manager
then holds only the on-connect snapshot, so "the bar before" is `None`, and
`execute_close_strategy` raised TypeError, silently killing the manager thread (HMC 2/04/25 12:52:00,
USO 4/30/25 11:42:00 in the harness).
- Fixed on orderPipe `feat/checkpoint-silence`: it stays on the ref stop until a prior bar exists.
- rule_sim mirrors the fix.
- Parity: quick 30/30, delayed 60/60.

## Deviations from the approved plan (2026-09-29)
- **Volume feature** is `vol_decay` (post-entry volume decay), not a Polygon time-of-day baseline
  ratio. It is computable from the same 5s bars live and in history, with no ~2k Polygon calls.
- **Mongo trades after 6/08 are not in the population.** Only ~20 are news-tagged; Mongo is used
  for anchor calibration only.
- **Account-R uses today's 1R ($3k).** No per-date 1R vintage is available in backtester.

## Status
- [x] population, bar cache (Trillium 5s), rule_sim, shared features, parity 30/30
- [x] unit tests: `tests/unit/test_checkpoint_rule_sim.py`; orderPipe `tests/test_checkpoint_features.py`
- [x] full fetch 2026-09-29 16:05–16:31 ET (1,506 ticker-days; 2 missing: GBTC 2023-07-13, PTPI 2024-02-14)
- [x] amendment 1: delayed rule co-primary + `switch` policy; parity quick 30/30, delayed 60/60
- [x] panels: 2 primaries + 22 variants (the anchor_m30 pair crashed on an empty pre-entry window,
      was fixed in orderPipe d311f10 and rebuilt; variants are only consulted for Stage-B passes)
- [x] `analyze.py` → `data/checkpoint_study/report/` (verdict.json, candidates.csv, cells_train.csv)

## Result — 2026-09-29: **KILL**
Survivors, of 1,691 simulated (88 dropped for price mismatch, 2 for missing bars):

| Rule | 6 min | 17 min | 30 min |
|---|---|---|---|
| quick | 726 | 101 | 9 |
| delayed | 1,083 | 357 | 88 |

Eligible checkpoints: quick 6m; delayed 6m and 17m. 144 train cells.

**Stage A (train).** Only 4 of 144 cells had CI lower bound > 0, which is about what chance gives
(144 × 2.5% ≈ 3.6). Three became candidates, and all three are the same idea: the top tercile of
`giveback_R` at 6m → exit or tighten.

| Cell | Train n, mean | Test n, mean | Test 95% CI | p | Holm p |
|---|---|---|---|---|---|
| [quick] 6m giveback_R > 0.54 → exit_now | 166, +0.28R | 73, +0.09R | [−0.27, +0.42] | 0.29 | 0.54 |
| [quick] 6m giveback_R > 0.54 → trail_1m | 166, +0.16R | 73, +0.08R | [−0.20, +0.31] | 0.27 | 0.54 |
| [delayed] 6m giveback_R > 0.67 → exit_now | 249, +0.25R | 115, +0.21R | [−0.07, +0.47] | 0.075 | 0.22 |

**Stage B: 0 passes. Placebo: 0/20.** Decision: the checkpoint stays orderPipe's neutral timer.

**Descriptive (post-verdict, not a finding).**
- **Every action is a wash on average.** Unconditionally (train), exit / tighten / switch vs hold are
  all within ±0.03R, with CIs about ±0.1R. Holding past the checkpoint adds ~0R on average under
  either rule.
- **Giveback is the only idea with a consistent sign.** Big giveback by 6m → get out kept a positive
  sign out of sample, but is unproven. Per §10, re-test only once the test population has grown ≥ 50%
  (≥ 834 test episodes; 556 now).

