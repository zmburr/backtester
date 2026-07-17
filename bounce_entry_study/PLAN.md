# Bounce Entry-Criteria Study — Design

**Research question:** Which mechanical entry signals — and which confluence combinations of them — recognize *the* bottom on bounce days with acceptable lag, acceptable fake-out rate, and positive entry economics?

**Why:** In the moment, single signals (a 2-min break, an apparent higher low, a trendline break) fake out constantly. Before wiring anything into the morning watcher, every candidate rule gets scored against the curated bounce book so "wait for confluence" becomes a number, not a feeling.

**Nature:** Descriptive signal study over ~100 curated days. No ML, no classifiers — the deliverable is a ranked table of detector configs with lag / fake-out / R distributions, plus per-day charts to eyeball the winners.

**Lives in:** `backtester/bounce_entry_study/` (mirrors `iv_study/` multi-file pattern). Bars cached to `data/bounce_entry_study/`, outputs to `reports/bounce_entry_study/`.

---

## 0. The core measurement problem, resolved up front

Every one of these signals fires **many times per day**. A study that asks "did it fire near the low?" will flatter everything. The honest framing:

- **The first actionable fire is the one you'd take.** For each config we evaluate the *first* fire of the session (after the time gate) — not the best one in hindsight.
- **A fake-out is a fire followed by a lower low.** If the signal fires and the stock later undercuts the fire-time session low (by > 0.10% — tick-noise epsilon), you were faked. That is exactly the lived experience being engineered away.
- **Economics beat timing.** A signal that fires 10 minutes "late" but never gets undercut can be worth more than one that nails the low 50% of the time. So every fire is also scored as a trade: entry at the fire bar's close, structural stop at the fire-time session low minus 0.25×ATR, and we record the R actually available afterward.

## 1. Data & cohorts

**Positive cohort — the bounce book.** All rows of `data/bounce_data.csv` (~100 days, 2018–2026). Per row we already have `date, ticker, Setup, cap, trade_grade, atr_pct, time_of_low_price, time_of_bounce, bounce_open_*` — used for truth-checking and for cutting results by setup type / cap / grade.

- Skip `.T` (Canadian) tickers — no clean historical minute source (~2–3 rows).
- Cross-check the CSV's `time_of_low_price` against the fetched bars; where they disagree by > 2 min, trust the bars and log the row.
- `IntradayCapitch` rows (17% WR pattern) stay in but are reported as their own cut — a bottom-detector that "works" mainly on those is a red flag.

**Control cohort — days that looked like bounces but weren't.** Run `scanners/bounce_backscanner.py` over 2023–2026, take candidate days that passed the pre-filter/scoring but (a) are not in the curated book and (b) closed red with a low in the final hour (the no-bounce failure mode). Target ~100 control days. The question it answers: *how often does "confluence" fire on a day where there was no bottom to find?* A config's fake-out rate on controls is its true cost.

**Bars.** `polygon_queries.get_intraday(ticker, date, 1, 'minute')` per (ticker, date), premarket included, cached to parquet keyed `{ticker}_{date}` — fetch once (~200 API calls total), rerun the grid for free. 1-min bars resampled to the 2-min frame all detectors run on. Session VWAP computed from 4:00 AM like the live watcher does.

## 2. Detector catalog (each pure, each parameterized)

All detectors consume the canonical 2-min RTH frame + ATR (from `atr_pct` × open, matching the live watcher's 9:30 lock) and return a list of fire timestamps with metadata. Pure functions — the exact code that later ports into a morning-watcher evaluator fed by live 2-min aggregation (the aggregation machinery already exists in `orderPipe/morning_watcher/rules/covering_rules.py`).

| ID | Detector | Definition | Parameters |
|----|----------|------------|------------|
| D1 | **2-min break up** | 2-min bar **closes** above prior 2-min bar's high. Optional HELD variant: next `h` bars don't take out the break bar's low (mirror of the short-side PBBTracker). | close-vs-wick basis; held-window `h ∈ {0, 2, 3}` |
| D2 | **Higher low** | Confirmed swing low (2-min bar low is minimum of `k` bars each side, confirmed `k` bars after the fact) that sits ≥ `f`×ATR above the prior swing low (first anchor = session low). | pivot width `k ∈ {1, 2, 3}`; floor `f ∈ {0, 0.10, 0.25}` |
| D3 | **Structure break** | 2-min close above the most recent *lower high* (descending swing highs tracked from the open, same pivot logic). The objective version of a downtrend-line break. | pivot width `k`; which lower high (most recent vs. highest of last 2) |
| D4 | **VWAP reclaim** | 2-min close above session VWAP after having spent ≥ `m` minutes below it. | `m ∈ {20, 40}` |
| D5 | **Down-vol dry-up** | Volume on down 2-min bars over the last `w` bars < `r`× the volume on down bars during the selloff leg (seller exhaustion). Supporting vote only — never a standalone entry. | `w ∈ {10, 20}`; `r ∈ {0.5, 0.7}` |

Time gate applied to all: no fires before `T0 ∈ {9:30, 9:45, 10:00}`. The 9:45/10:00 variants encode the cohort fact that 77% of bounce lows are in by 10:00.

## 3. Confluence engine

A config = a subset of detectors + `K`-of-`N` requirement + rolling window `W ∈ {15, 20, 30}` minutes + time gate `T0`. Confluence fires at the first bar where ≥ K distinct detectors have fired within the trailing W minutes. Two structural variants worth testing beyond plain K-of-N:

- **Ordered:** D2 (higher low) must be the *latest* of the contributing fires — confluence confirmed by structure, not by momentum alone.
- **Reset-on-new-low:** any new session low clears the accumulated fires (a break that precedes a flush shouldn't count toward the confluence that follows).

Grid size: 5 detector subsets of interest (D1+D2, D1+D3, D2+D3, D1+D2+D3, all-5 with K=3) × parameter combos ≈ **300–500 configs** × ~200 days — pure pandas on cached bars, minutes to run.

## 4. Metrics

Per (day, config): first-fire time; **lag** = fire − true-low time (negative = fired before the low); **faked** = later undercut of fire-time session low; **miss** = never fired; entry economics from fire-bar close with stop = fire-time low − 0.25×ATR → **R at close**, **max R before close** (day-high after fire), stopped-out flag.

Aggregates per config, on positives: miss %, fake-out %, median/IQR lag, median R-at-close, % of days R ≥ 1. On controls: fire % (lower is better), median R of those fires (how much the fakes cost). Cut by setup type (weakstock vs strongstock vs IntradayCapitch), cap, grade, and **by year** — 2018–2026 spans regimes, and a config whose edge lives entirely in one year is noise. Prefer the *simplest* config within noise of the best (fewest detectors, loosest parameters).

## 5. Deliverables

```
bounce_entry_study/
├── PLAN.md            # this file
├── fetch_bars.py      # cache 1-min bars per (ticker, date) → data/bounce_entry_study/
├── detectors.py       # D1–D5 pure functions (the port target for the live watcher)
├── confluence.py      # K-of-N engine + ordered / reset variants
├── run_study.py       # grid → per-(day, config) rows → metrics parquet
├── report.py          # ranked config table (CSV) + per-day chart PNGs for top configs
└── build_controls.py  # backscanner-driven control-day selection
```

Per-day charts for the top ~5 configs: 2-min candles, true low marked, each detector fire, confluence fire, stop line — the eyeball check that catches metric-gaming before it ships.

## 6. Systemization gate (the point of all this)

A config earns a live evaluator only if, roughly: fake-out ≤ ~20% on positives, control fire-rate materially below positive fire-rate, median lag ≤ ~15 min, median entry R ≥ 1 — thresholds to be finalized *after* seeing the distributions, but written down **before** porting so the decision isn't vibes. Porting is then: `detectors.py` verbatim into a `bottom_structure_rules.py` evaluator in the morning watcher, confluence auto-flips the `remind_confluence` checklist row, 3-of-3 earns TTS. Playback replays of curated days become the integration test.

## 7. Decisions (user, 2026-07-17)

1. **Entry**: at fire time and price exactly — the fire bar's close, no retest limit.
2. **Stop**: the raw low of day (session low at fire time). No ATR buffer.
3. **In-day stop-outs on true bounce days are expected** — a signal can trigger, break lows, stop the entry out, and the stock still bounces later. Therefore the simulation models **re-entries**: after a stop-out, the next fresh confluence fire (post-reset on the new low) enters again. A config is scored by **total day R across attempts** (each attempt risks 1R; stopped = −1R, survivor = (close − entry)/risk), matching the live 1R framework.
4. **Cohort**: v1 runs on the curated bounce book only ("the examples I have"), today's rows included — today (7/17 DRAM/SOXL/CRDO) reported separately as a partial-day demo, excluded from aggregates. Control cohort deferred to v2.
5. R normalization guard: risk floored at 0.25% of entry so a fire bar sitting on the session low doesn't produce absurd R.
