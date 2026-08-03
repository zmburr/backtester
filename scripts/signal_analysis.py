"""Signal Analysis Feedback Loop v2 — periodic deep analysis of scorecard outcomes.

Replaces dispatcher.signal_analysis. Differences from v1:
- Reads the richer multi-day outcomes CSV (data/signal_outcomes.csv) produced by
  scripts/signal_scorecard.py, including earliness metrics (days_to_1atr,
  adverse_before_fav_atr) so the analysis can evaluate "scanner fires early".
- Remembers its own past recommendations (state file) and feeds them back into
  the prompt so successive analyses build on each other instead of re-churning.
- Statistical guardrails in the prompt: no threshold recommendation from cells
  with n < 20; pooled-cap analysis preferred over per-cap at current sample sizes.
- Recommendations land in the email report only (no Todoist — the v1 batch API
  endpoint was returning 410s anyway).

Usage:
    python scripts/signal_analysis.py            # gated: runs every 15 new complete signals
    python scripts/signal_analysis.py --force    # ignore the gate
    python scripts/signal_analysis.py --dry      # print report, no email/state update
"""

import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from support.config import send_email  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("signal_analysis")

OUTCOMES_FILE = PROJECT_ROOT / "data" / "signal_outcomes.csv"
STATE_FILE = PROJECT_ROOT / "data" / "signal_analysis_state.json"
EMAIL_TO = "zmburr@gmail.com"
SIGNAL_GATE = 15
CLAUDE_BIN = "/opt/homebrew/bin/claude"
MIN_CELL_N = 20
EPISODE_GAP = 3  # keep in sync with signal_scorecard.EPISODE_GAP
CLUSTER_DAY_THRESHOLD = 15  # keep in sync with priority_report.CLUSTER_DAY_THRESHOLD


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"analysis_number": 0, "signal_count_at_last_analysis": 0, "past_recommendations": []}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_complete_rows() -> list[dict]:
    if not OUTCOMES_FILE.exists():
        return []
    with open(OUTCOMES_FILE, newline="") as f:
        return [r for r in csv.DictReader(f)
                if str(r.get("complete")).lower() == "true"
                and str(r.get("days_available") or "0") != "0"]  # skip no-data (delisted) rows


def _extract_section(filepath: Path, start_marker: str, max_lines: int) -> str:
    if not filepath.exists():
        return f"({filepath.name} not found)"
    text = filepath.read_text(encoding="utf-8", errors="replace")
    idx = text.find(start_marker)
    if idx == -1:
        return f"(marker '{start_marker}' not found in {filepath.name})"
    return "\n".join(text[idx:].splitlines()[:max_lines])


MAX_PROMPT_ROWS = 400  # keep the prompt bounded as the log grows

COHORT_CONTROL = "control"
COHORT_SURFACED = "surfaced"


def _cohort(row: dict) -> str:
    return (row.get("cohort") or COHORT_SURFACED).strip() or COHORT_SURFACED


def _score_num(row: dict):
    """Numeric score out of the 'N/M' score string, or None."""
    head = str(row.get("score") or "").split("/", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def _rate(rows: list[dict]) -> dict | None:
    """tradeable_3d pooled AND cluster-weighted, with the cluster count.

    Cluster-weighting averages within each cluster_id first, then averages those
    means — so a 19-signal single-session wave counts once, not nineteen times.
    """
    if not rows:
        return None
    hits = sum(1 for r in rows if str(r.get("tradeable_3d")).lower() == "true")
    by_cluster: dict[str, list] = {}
    for r in rows:
        by_cluster.setdefault(r.get("cluster_id") or "", []).append(r)
    means = [sum(1 for x in g if str(x.get("tradeable_3d")).lower() == "true") / len(g)
             for g in by_cluster.values()]
    return {
        "n": len(rows), "hits": hits,
        "pooled": round(hits / len(rows) * 100, 1),
        "cw": round(sum(means) / len(means) * 100, 1),
        "clusters": len(by_cluster),
    }


def _fmt_rate(label: str, r: dict | None) -> str:
    if not r:
        return f"{label:<34} (no rows)"
    return (f"{label:<34} {r['hits']:>4}/{r['n']:<5} = {r['pooled']:>5.1f}%   "
            f"cluster-weighted {r['cw']:>5.1f}%   ({r['clusters']} clusters)")


def _paired_sessions(rows: list[dict], bucket: str) -> list[tuple]:
    """Per-session (surfaced_rate, control_rate) pairs for one bucket.

    Sessions where either cohort is absent are dropped — an unpaired session
    contributes nothing to a within-session comparison.
    """
    by_sess: dict[str, dict] = {}
    for r in rows:
        if r.get("bucket") != bucket:
            continue
        slot = by_sess.setdefault(r.get("target_date", ""), {COHORT_SURFACED: [], COHORT_CONTROL: []})
        slot[_cohort(r)].append(r)

    def _hit_rate(g):
        return sum(1 for x in g if str(x.get("tradeable_3d")).lower() == "true") / len(g)

    out = []
    for date, slot in sorted(by_sess.items()):
        s, c = slot[COHORT_SURFACED], slot[COHORT_CONTROL]
        if s and c:
            out.append((date, _hit_rate(s), len(s), _hit_rate(c), len(c)))
    return out


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def _entry_r_block(bucket_rows: list[dict]) -> list[str]:
    """Entry-anchored magnitude, replayed through the live 2-min entry rule.

    entry_r is a CONTINUOUS scale (MFE in R off the real LOD stop), not a
    binary — a bigger bounce scores higher. It is the only measure here anchored
    at a price the rules would actually have filled at, so it is the one to
    trust when it disagrees with the open-anchored tradeable_3d.
    """
    def er(r):
        try:
            return float(r["entry_r"])
        except (TypeError, ValueError, KeyError):
            return None

    scored = [(r, er(r)) for r in bucket_rows]
    scored = [(r, v) for r, v in scored if v is not None]
    if not scored:
        return []

    out = ["  ENTRY-ANCHORED MAGNITUDE (live 2-min break rule, D0, MFE in R off the LOD stop)"]
    for label, sel in (("surfaced", COHORT_SURFACED), ("control", COHORT_CONTROL)):
        vals = [v for r, v in scored if _cohort(r) == sel]
        if vals:
            out.append(f"    {label:<10} n={len(vals):<5} median {_median(vals):>5.2f}R   "
                       f"mean {sum(vals)/len(vals):>5.2f}R   "
                       f">=1R {sum(1 for v in vals if v >= 1)/len(vals)*100:>4.1f}%")

    # paired within session, on medians
    by: dict[str, dict] = {}
    for r, v in scored:
        slot = by.setdefault(r.get("target_date", ""), {COHORT_SURFACED: [], COHORT_CONTROL: []})
        slot[_cohort(r)].append(v)
    diffs = [_median(s[COHORT_SURFACED]) - _median(s[COHORT_CONTROL])
             for s in by.values() if s[COHORT_SURFACED] and s[COHORT_CONTROL]]
    if diffs:
        out.append(f"    PAIRED per-session edge: mean {sum(diffs)/len(diffs):+.2f}R   "
                   f"median {_median(diffs):+.2f}R   surfaced won "
                   f"{sum(1 for d in diffs if d > 0)}/{len(diffs)} sessions")

    out.append("    by score (median entry_r):")
    for s in range(7):
        vals = [v for r, v in scored if _score_num(r) == s]
        if vals:
            out.append(f"      score {s}: n={len(vals):<5} median {_median(vals):>5.2f}R   "
                       f">=1R {sum(1 for v in vals if v >= 1)/len(vals)*100:>4.1f}%")
    return out


def _breadth_block(bucket_rows: list[dict]) -> list[str]:
    """Realised R by same-session breadth (how many surfaced windows fired).

    This is the table that recalibrates CLUSTER_DAY_THRESHOLD. It reports
    SESSIONS, not rows, as the effective n — a band can show 168 rows off two
    sessions, which is how a "+1.72R" market-wide figure drawn from 4 sessions
    read as solid when it was not. Never quote a band's R without its sessions.
    """
    def xr(r):
        try:
            return float(r["exit_r"])
        except (TypeError, ValueError, KeyError):
            return None

    surfaced_per_session: dict[str, int] = {}
    for r in bucket_rows:
        if _cohort(r) == COHORT_SURFACED:
            d = r.get("target_date", "")
            surfaced_per_session[d] = surfaced_per_session.get(d, 0) + 1

    scored = [(r, xr(r)) for r in bucket_rows]
    scored = [(r, v) for r, v in scored if v is not None]
    if not scored:
        return []

    out = ["  BREADTH -> REALISED R (windows open that session; sessions = effective n)"]
    for lo, hi, lab in ((0, 2, "0-2"), (3, 6, "3-6"), (7, 14, "7-14"), (15, 10 ** 9, "15+")):
        sub = [(r, v) for r, v in scored
               if lo <= surfaced_per_session.get(r.get("target_date", ""), 0) <= hi]
        if not sub:
            continue
        vals = [v for _, v in sub]
        sess = len({r.get("target_date") for r, _ in sub})
        wins = sum(1 for v in vals if v > 0)
        out.append(f"    {lab:<6} sessions {sess:>3}  rows {len(vals):>4}  "
                   f"EV {sum(vals)/len(vals):>+6.2f}R  win {wins/len(vals)*100:>5.1f}%")
    return out


def build_cohort_block(rows: list[dict]) -> str:
    """Pre-computed entry-bar tables.

    Control rows are NOT sent as raw CSV: they outnumber surfaced rows ~10:1 and
    would consume the whole MAX_PROMPT_ROWS budget. Aggregating in Python also
    removes an arithmetic failure mode — the model reads conclusions off exact
    counts instead of summing hundreds of rows by hand.

    Two deliberate design choices, both of which the prompt explains:

    1. Control rates are computed over ALL rows, not first-flags. Episode
       chaining assumes a re-flag double-counts one move; a ticker that sits
       below the bar for a month instead yields exactly ONE first-flag, dated to
       whenever the observation window happened to open. On the 2026-08-03
       backfill, 70 of 127 control first-flags landed on the window's opening
       day. First-flag filtering is right for surfaced signals and meaningless
       for a persistent cohort.
    2. The headline test is PAIRED WITHIN SESSION. Surfaced signals concentrate
       on days when everything worked, so an unpaired pooled comparison mostly
       measures which days each cohort appeared on, not whether the bar selects.
    """
    out: list[str] = []

    for bucket in ("bounce", "reversal"):
        b = [r for r in rows if r.get("bucket") == bucket]
        if not b:
            continue
        ctrl = [r for r in b if _cohort(r) == COHORT_CONTROL]
        surf = [r for r in b if _cohort(r) == COHORT_SURFACED]
        out.append(f"\n{bucket.upper()}")

        if not ctrl:
            out.append("  No control rows yet for this bucket — the entry bar is NOT testable here.")
            out.append(_fmt_rate("  surfaced first-flags", _rate(
                [r for r in surf if str(r.get("episode_signal_num", "")) == "1"])))
            continue

        # -- Headline: paired within-session comparison -----------------------
        pairs = _paired_sessions(b, bucket)
        out.append(f"  PAIRED WITHIN-SESSION TEST ({len(pairs)} sessions with both cohorts present)")
        if pairs:
            diffs = [s - c for _, s, _, c, _ in pairs]
            wins = sum(1 for d in diffs if d > 0)
            losses = sum(1 for d in diffs if d < 0)
            mean_d = sum(diffs) / len(diffs)
            med_d = sorted(diffs)[len(diffs) // 2]
            out.append(f"    mean per-session edge (surfaced - control): {mean_d * 100:+.1f} pp")
            out.append(f"    median per-session edge:                    {med_d * 100:+.1f} pp")
            out.append(f"    sessions surfaced won / lost:               {wins} / {losses}")
            out.append("    per-session detail (surfaced hits/n vs control hits/n):")
            for date, sr, sn, cr, cn in pairs:
                out.append(f"      {date}  surfaced {sr * 100:>5.1f}% (n={sn:<3})  "
                           f"control {cr * 100:>5.1f}% (n={cn:<3})  {(sr - cr) * 100:>+6.1f} pp")

        # -- Score curve: does tradeable_3d rise with score? ------------------
        out.append("  SCORE CURVE (all rows, cluster-weighted by session)")
        for s in range(7):
            sub = [r for r in b if _score_num(r) == s]
            if not sub:
                continue
            cohorts = "+".join(sorted({_cohort(r) for r in sub}))
            out.append(_fmt_rate(f"    score {s} [{cohorts}]", _rate(sub)))
        out.append(_fmt_rate("    BELOW BAR (score <= 3)",
                             _rate([r for r in b if (_score_num(r) if _score_num(r) is not None else 99) <= 3])))
        out.append(_fmt_rate("    AT/ABOVE BAR (score >= 4)",
                             _rate([r for r in b if (_score_num(r) if _score_num(r) is not None else -1) >= 4])))

        # -- Surfaced first-flags, unchanged, for continuity with prior cycles
        out.append(_fmt_rate("    surfaced first-flags (as in prior analyses)", _rate(
            [r for r in surf if str(r.get("episode_signal_num", "")) == "1"])))

        # -- Entry-anchored magnitude (the trader's actual rule) --------------
        block = _entry_r_block(b)
        if block:
            out.extend(block)

        # -- Breadth: the table that recalibrates the cluster-day threshold ---
        block = _breadth_block(b)
        if block:
            out.extend(block)

    if not out:
        return "(no rows scored yet — entry-bar test not yet possible)"
    return "\n".join(out)


def build_prompt(rows: list[dict], state: dict) -> str:
    analysis_num = state.get("analysis_number", 0) + 1

    # The raw table stays surfaced-only so the existing analysis is unchanged and
    # the row budget is not eaten by control rows; control enters via aggregates.
    surfaced = [r for r in rows if _cohort(r) == COHORT_SURFACED]
    control = [r for r in rows if _cohort(r) == COHORT_CONTROL]

    recent = sorted(surfaced, key=lambda r: r.get("target_date", ""))[-MAX_PROMPT_ROWS:]
    omitted = len(surfaced) - len(recent)
    header = list(recent[0].keys()) if recent else []
    table = ",".join(header) + "\n"
    for r in recent:
        table += ",".join(str(r.get(c, "")) for c in header) + "\n"
    if omitted:
        table += f"\n({omitted} older signals omitted — totals above reflect the full log)\n"

    cohort_block = build_cohort_block(rows)

    past = state.get("past_recommendations", [])
    past_text = "None — this is the first analysis on the v2 multi-day data." if not past else "\n".join(
        f"- (analysis #{p['analysis']}) {p['text']}" for p in past[-20:]
    )

    rev_thresholds = _extract_section(PROJECT_ROOT / "analyzers" / "reversal_scorer.py", "CAP_THRESHOLDS", 60)
    bounce_thresholds = _extract_section(PROJECT_ROOT / "analyzers" / "bounce_scorer.py", "SETUP_PROFILES", 120)

    return f"""You are a quantitative trading systems analyst reviewing scanner signal performance.
This is Analysis #{analysis_num} with {len(rows)} fully-scored signals (each scored over a D0..D+3 trading-day window): {len(surfaced)} surfaced (cleared the entry bar) and {len(control)} control (below-bar counterfactual).

## How signals are scored
- bucket=reversal means SHORT thesis (favorable = down); bucket=bounce means LONG (favorable = up).
- There is no VETO cohort: the prior_day_rvol < 1.25 hard veto from Analysis #2 shipped 2026-06-10 and was removed again on 2026-07-30 (commit 32224a7e) without ever firing — it never emitted a single row, because the lowest prior_day_rvol on any post-deploy reversal was 1.286. Do NOT ask for the vetoed cohort's rate and do NOT re-derive the veto; that question is closed as untestable on this data.
- cohort: 'surfaced' rows cleared the entry bar (recommendation GO or CAUTION, i.e. score >= 4) and are the signals actually acted on. 'control' rows are the below-bar NO-GO signals from the SAME watchlist scan, scored over the same D0..D+3 window purely as a counterfactual. Added 2026-08-03; before that the log contained surfaced rows only, which is why earlier analyses could compare GO vs CAUTION but never test the bar itself.
- Episodes and clusters are numbered WITHIN cohort, so episode_id/cluster_id never mix the two. Never compare a surfaced row to a control row via episode_id, and never pool the cohorts into a single headline rate — control rows were not traded and are not a performance number. Use them only for the entry-bar question.
- The signal-outcomes table below is SURFACED ROWS ONLY. The control cohort is supplied pre-aggregated in the 'Entry-bar cohorts' section, because it outnumbers the surfaced rows by ~10:1 and would otherwise crowd out the raw table. Treat those counts as exact — do not attempt to re-derive them.
- d0_pct..d3_pct: cumulative close-vs-entry-open raw price move per day.
- mfe_atr_3d / mae_atr_3d: max favorable / adverse excursion over the window in ATRs.
- tradeable_3d: MFE hit the per-bucket gate at any point in the window (primary success metric). Bounce gate = min(0.5 x ATR, 6% absolute) — the playbook T1 target, capped because signal-day ATR is inflated by the selloff itself. Reversal gate = 1.0 x ATR.
- days_to_1atr: trading days until the deeper 1-ATR target hit ('' = never) — note this is NOT the tradeable gate for bounces.
- adverse_before_fav_atr: worst adverse run (ATRs) BEFORE the favorable target hit — this measures how EARLY the scanner fires. Large values on reversals mean the stock kept squeezing up before cracking.
- Pre-trade criterion features logged per signal (values at alert time): reversal signals carry pct_from_9ema, pct_change_3, gap_pct, prior_day_range_atr, prior_day_rvol, premarket_rvol; bounce signals carry selloff_total_pct, pct_off_30d_high, pct_off_52wk_high, pct_change_3, gap_pct, prior_day_range_atr. Use these for criterion-level threshold analysis: compare feature distributions of tradeable vs non-tradeable signals (first-flags only) and look for cut points that would have filtered losers without dropping winners. Blank = the report didn't emit that metric for that signal.
- episode_id / episode_signal_num: repeat flags of the same ticker within {EPISODE_GAP} trading days chain into one episode; their outcome windows OVERLAP, so rows within an episode are NOT independent samples. For any statistical claim, use episode_signal_num == 1 rows (or count distinct episode_id) as the sample. Reprints (episode_signal_num >= 2) may be analyzed separately as a persistence feature — "does a 2nd/3rd consecutive flag predict better odds?" — but never mix them into per-signal rates as if independent.
- cluster_id / cluster_size: the SECOND correlation axis, orthogonal to episodes. cluster_id = (bucket, target_date); cluster_size = how many distinct first-flags fired in that same session. Ten tickers flagging off one sector move share a single market event, so they are closer to one observation than ten — 2026-07-17 alone produced 13 bounce first-flags that ALL resolved tradeable. Episode chaining does not catch this. Whenever you report a pooled rate, also report the cluster-weighted rate (average within each cluster_id, then average those cluster means) and say which one you are drawing the conclusion from. If a cell's result is carried by one or two large clusters, say so explicitly and treat its effective n as the number of clusters, not the number of rows.
- mfe_atr_3d / mae_atr_3d are stored rounded. Rows written before 2026-07-31 carry 2 decimals, so a displayed "1.0" can be a raw 0.997 — that is why some rows show mfe_atr_3d = 1.00 with a blank days_to_1atr. This is a display artifact, NOT a writer bug (the days_to_1atr comparison has always been >=). Do not raise it as a data-QA item.

## Signal outcomes (complete windows only) — SURFACED cohort
{table}

## Entry-bar cohorts (pre-aggregated, exact counts — surfaced + control)
{cohort_block}

## Current thresholds (read-only context)
Reversal CAP_THRESHOLDS (score >= 4 GO, == 3 CAUTION):
{rev_thresholds}

Bounce SETUP_PROFILES (score >= 5 GO, == 4 CAUTION):
{bounce_thresholds}

## Recommendations from previous analyses (do not repeat unless new data strengthens or reverses them)
{past_text}

## Settled questions — do not relitigate without new contradicting data
(This list is the record. The former docs/signal_findings.md ledger was deleted on 2026-07-30 in commit 32224a7e; do not cite or ask for that file.)
- D0-close-direction confirmation: REJECTED. Long-first tactics: REJECTED. Entry-delay variants: CLOSED.
- 1.5-ATR initial stop: AFFIRMED.
- RVOL >= 1.25 veto: REMOVED 2026-07-30, never fired. Closed — see the recommendation-field note above.
- gap_pct -> RVOL-tier score restructure: TESTED 2026-06-10, scored WORSE than the plain veto (18/33 = 54.5% vs 59.2%) — deferred until ~100 first-flags.
- Reversal 5/5-vs-4/5 score inversion: CLOSED in Analysis #7 as a cap-mix confound, then wrongly reopened in #12 and #13. It is a Medium-cap artifact (Medium 4/5 runs 19/23 while Large is ~32% at every tier); within cap there is no inversion. Report score tiers cap-stratified and stop proposing a 5-point-score restructure on the pooled figure.
- Reversal signal drought from 2026-07 on: EXPLAINED, not a defect. The market is in a broad decline, so almost nothing sets up as a parabolic short and the router sends candidates to the bounce bucket instead. State the reversal first-flag count for the cycle and move on — do not open it as a finding, and do not treat a stalled reversal counter as a reason to defer other work.
- Bounce GO-vs-CAUTION separation: CLOSED as flat after three consecutive retests (#12, #13, #14). Do not re-run it. The open question is now the ENTRY BAR itself (score >= 4 vs <= 3), which the control cohort finally makes testable — that is a different question and it is NOT settled.
- Control cohort provenance: backfilled 2026-08-03 from signal_ledger.csv for sessions 2026-07-09 onward. Backfilled control rows carry atr_pct, gap_pct, pct_change_3, prior_day_range_atr, pct_from_9ema, prior_day_rvol and premarket_rvol, but NOT the bounce depth features (selloff_total_pct, pct_off_30d_high, pct_off_52wk_high) — the ledger never stored those. So depth-criterion cuts remain surfaced-only until enough forward-logged control rows accumulate; say "control lacks this feature" rather than treating the blanks as data. The ledger holds no reversal rows for that period (the drought), so the reversal control cohort starts empty and fills going forward.

## Statistical guardrails — follow strictly
- Do NOT recommend a threshold change based on any cell (bucket x cap x criterion) with n < {MIN_CELL_N}. Say "insufficient sample" instead.
- Prefer pooled-across-cap conclusions at current sample sizes, EXCEPT where cap is itself the variable under test — reversal outcomes differ sharply by cap, so any reversal score-tier claim must be cap-stratified.
- Count clusters, not rows, when judging whether a cell is really at n >= {MIN_CELL_N}. A cell of 20 first-flags drawn from 3 cluster_ids is 3 observations wearing a costume; say so rather than clearing the gate on the row count.
- When you cite a rate, include the count (e.g. "4/19"). Note that a 30% vs 50% difference on n<30 is usually noise.
- Distinguish "the scanner is wrong" from "the scanner is early": use days_to_1atr and adverse_before_fav_atr.

## Output format
### PERFORMANCE SUMMARY
Give per-bucket first-flag tradeable_3d both ways: pooled (x/y) and cluster-weighted, plus the number of distinct cluster_ids behind each. Name any cluster contributing more than a quarter of a bucket's first-flags.
### EARLINESS ANALYSIS
Is the reversal scanner early? Quantify using days_to_1atr and adverse_before_fav_atr. Would waiting for a confirmation trigger (or a long-first tactic) have helped, based on this data?
### ENTRY BAR
Answer one question per bucket: does clearing the entry bar (score >= 4) actually predict a better outcome than failing it?
- Draw your verdict from the PAIRED WITHIN-SESSION TEST, not from any pooled surfaced-vs-control number. Surfaced signals concentrate on days when everything worked, so a pooled comparison largely measures which days each cohort appeared on. Report the mean and median per-session edge and the win/loss session split, and treat the number of PAIRED SESSIONS as the effective n — if it is below {MIN_CELL_N}, say the question remains open and report the running numbers without a verdict.
- A split near half the sessions is a coin flip no matter how large the pp gap looks; say so explicitly rather than reporting the gap alone.
- Also report whether the rate is MONOTONIC in score across the SCORE CURVE table. A bar that selects should show tradeable_3d rising with score. If it is flat, say plainly that the score is not selecting and that its only defensible role is as a quantity limiter — then name which individual criteria DO separate, since a flat total built from two good criteria and four noisy ones is a weighting problem, not proof the features are worthless.
- Do NOT apply first-flag filtering to control rows or cite a control first-flag count. A ticker that sits below the bar for weeks produces one first-flag dated to whenever the observation window opened; the control tables above are deliberately computed over all rows for this reason.
- Weigh the ENTRY-ANCHORED MAGNITUDE table most heavily. tradeable_3d measures MFE from the D0 OPEN, which structurally under-credits the core setup: a gap-down that flushes BELOW the open and reverses scores negative even when buying the flush paid. entry_r replays the trader's live 2-min prior-bar-break rule (LOD-recency gate, stop at the low of day) and measures MFE in R from the price that rule would actually have filled at. When entry_r and tradeable_3d disagree, entry_r is the truer read. Report entry_r as a distribution (median, >=1R share), never as a pass/fail rate — it is deliberately a continuous magnitude scale so a bigger bounce rates higher.
- entry_r caveats to state whenever you cite it: (a) it is MAXIMUM FAVOURABLE EXCURSION, not realised P&L — capturing it still requires an exit rule, so never present median entry_r as expected profit; (b) it is D0-only, and this scanner is known to fire early, so signals that set up on D+1/D+2 are scored as though they produced nothing; (c) no slippage or commission. entry_r_floored applies a 0.5-ATR risk floor, but signal-day atr_pct is inflated by the selloff (median ~16%), so that floor imposes an unrealistically large minimum risk and reads far lower — prefer entry_r and mention the floored figure only as a conservative bound.
### BREADTH / CLUSTER DAY
Read the BREADTH -> REALISED R table. The report's CLUSTER DAY banner fires at {CLUSTER_DAY_THRESHOLD} surfaced bounce windows; it was raised from 7 on 2026-08-03 because the 7-14 band measured -0.50R across 10 sessions while firing "outsized opportunity day, lean aggressive". State each band's EV with its SESSION count, never rows alone. If the 15+ band has reached {MIN_CELL_N} sessions, say the threshold can now be recalibrated with real power and give the number; if it has not, say how many sessions it stands at and explicitly withhold a recommendation. Flag it if any band's sign has flipped versus the table in the priority_report comment.
### CRITERIA EFFECTIVENESS
### THRESHOLD RECOMMENDATIONS
For each (only if guardrails allow), one line:
THRESHOLD_CHANGE: <bucket> | <cap or POOLED> | <criterion> | <current> | <recommended> | <evidence with counts>
### ACTION ITEMS
For each actionable follow-up, one line:
ACTION_ITEM: <concise description>"""


ROUTING_FILE = "/Users/zacharyburr/PycharmProjects/dispatcher/model_routing.json"
USAGE_LOG = os.path.expanduser("~/logs/claude_usage.jsonl")
JOB_NAME = "signal_analysis"
FALLBACK_MODEL = "claude-sonnet-5"


def _resolve_model() -> str:
    """Read the shared model_routing.json; never allow fable/mythos on API billing."""
    try:
        with open(ROUTING_FILE) as f:
            routing = json.load(f)
        route = routing.get("jobs", {}).get(JOB_NAME, routing.get("default", {}))
        model = route.get("model", FALLBACK_MODEL)
        banned = routing.get("banned_model_substrings", ["fable", "mythos"])
        if any(b in model.lower() for b in banned):
            model = FALLBACK_MODEL
        return model
    except (OSError, json.JSONDecodeError):
        return FALLBACK_MODEL


def _log_usage(record: dict) -> None:
    try:
        os.makedirs(os.path.dirname(USAGE_LOG), exist_ok=True)
        with open(USAGE_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.warning(f"Could not write usage log: {e}")


def run_claude(prompt: str) -> str:
    """Run claude -p in its own process group so a timeout kills MCP children too."""
    import signal as _signal
    from datetime import timezone

    model = _resolve_model()
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    started = time.time()
    try:
        proc = subprocess.Popen(
            [CLAUDE_BIN, "-p", prompt, "--permission-mode", "bypassPermissions",
             "--model", model, "--output-format", "stream-json", "--verbose"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env, cwd="/tmp", start_new_session=True,
        )
    except FileNotFoundError:
        logger.error(f"Claude CLI not found at {CLAUDE_BIN}")
        return ""
    try:
        stdout, stderr = proc.communicate(timeout=1800)
        if proc.returncode != 0:
            logger.error(f"Claude CLI exited {proc.returncode}: {(stderr or '')[:300]}")
    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timed out after 30 minutes — killing process group")
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
        except Exception:
            proc.kill()
        return ""

    # Parse the stream-json events: the final "result" event carries text + usage
    result_event = None
    text_parts = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result_event = event
        elif event.get("type") == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

    u = (result_event or {}).get("usage", {})
    _log_usage({
        "ts": datetime.now(timezone.utc).isoformat(),
        "job": JOB_NAME,
        "provider": "claude-cli",
        "model": model,
        "input_tokens": u.get("input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "cache_read_tokens": u.get("cache_read_input_tokens", 0),
        "cache_write_tokens": u.get("cache_creation_input_tokens", 0),
        "cost_usd": (result_event or {}).get("total_cost_usd"),
        "num_turns": (result_event or {}).get("num_turns"),
        "session_id": (result_event or {}).get("session_id"),
        "duration_s": round(time.time() - started, 1),
        "complete": result_event is not None,
    })

    if result_event and result_event.get("result"):
        return result_event["result"].strip()
    return "\n\n".join(p for p in text_parts if p).strip()


def parse_lines(output: str, tag: str) -> list[str]:
    return [m.group(1).strip() for line in output.splitlines()
            if (m := re.match(rf"{tag}:\s*(.+)", line.strip()))]


def format_email(report: str, n_signals: int, analysis_num: int) -> str:
    try:
        import markdown
        body = markdown.markdown(report, extensions=["tables", "fenced_code"])
    except ImportError:
        body = f"<pre style='white-space:pre-wrap;'>{report}</pre>"
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background-color:#0a0c10;color:#c8cdd8;padding:20px;max-width:860px;margin:0 auto;">
  <div style="border-bottom:2px solid #3b82f6;padding-bottom:12px;margin-bottom:20px;">
    <h1 style="color:#e8ecf4;font-size:20px;margin:0 0 6px 0;">Signal Analysis v2 &mdash; #{analysis_num}</h1>
    <span style="color:#6b7280;font-size:13px;">{now} &mdash; {n_signals} complete signals</span>
  </div>
  <div style="color:#c8cdd8;font-size:14px;line-height:1.7;">{body}</div>
  <div style="border-top:1px solid #1e2330;margin-top:30px;padding-top:12px;color:#4b5563;font-size:11px;">
    Signal Analysis v2 &mdash; Backtester
  </div>
</body></html>"""


def main():
    dry = "--dry" in sys.argv
    force = "--force" in sys.argv
    logger.info(f"Signal Analysis v2 starting{' (dry)' if dry else ''}{' (forced)' if force else ''}...")

    rows = load_complete_rows()
    state = load_state()
    if not rows:
        logger.info("No complete signals in the outcomes log yet. Skipping.")
        return
    new = len(rows) - state.get("signal_count_at_last_analysis", 0)
    logger.info(f"{len(rows)} complete signals, {new} new since last analysis")
    if new < SIGNAL_GATE and not force:
        logger.info(f"Need {SIGNAL_GATE} new signals to trigger. Skipping.")
        return

    prompt = build_prompt(rows, state)
    logger.info(f"Prompt built ({len(prompt)} chars). Running Claude...")
    report = run_claude(prompt)
    if not report:
        logger.error("No output from Claude. Aborting.")
        sys.exit(1)

    changes = parse_lines(report, "THRESHOLD_CHANGE")
    actions = parse_lines(report, "ACTION_ITEM")
    analysis_num = state.get("analysis_number", 0) + 1
    logger.info(f"Analysis #{analysis_num}: {len(changes)} threshold recs, {len(actions)} action items")

    if dry:
        print(report)
        return

    state["analysis_number"] = analysis_num
    state["signal_count_at_last_analysis"] = len(rows)
    state["last_analysis_at"] = datetime.now().strftime("%Y-%m-%d")
    recs = state.setdefault("past_recommendations", [])
    recs.extend({"analysis": analysis_num, "text": t} for t in changes + actions)
    state["past_recommendations"] = recs[-60:]
    save_state(state)

    html = format_email(report, len(rows), analysis_num)
    subject = f"Signal Analysis #{analysis_num} — {len(rows)} signals, {len(changes)} recommendations"
    send_email(EMAIL_TO, subject, html, is_html=True)
    logger.info("Analysis email sent. Done.")


if __name__ == "__main__":
    main()
