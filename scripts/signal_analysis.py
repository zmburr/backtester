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


def build_prompt(rows: list[dict], state: dict) -> str:
    analysis_num = state.get("analysis_number", 0) + 1

    recent = sorted(rows, key=lambda r: r.get("target_date", ""))[-MAX_PROMPT_ROWS:]
    omitted = len(rows) - len(recent)
    header = list(recent[0].keys())
    table = ",".join(header) + "\n"
    for r in recent:
        table += ",".join(str(r.get(c, "")) for c in header) + "\n"
    if omitted:
        table += f"\n({omitted} older signals omitted — totals above reflect the full log)\n"

    past = state.get("past_recommendations", [])
    past_text = "None — this is the first analysis on the v2 multi-day data." if not past else "\n".join(
        f"- (analysis #{p['analysis']}) {p['text']}" for p in past[-20:]
    )

    rev_thresholds = _extract_section(PROJECT_ROOT / "analyzers" / "reversal_scorer.py", "CAP_THRESHOLDS", 60)
    bounce_thresholds = _extract_section(PROJECT_ROOT / "analyzers" / "bounce_scorer.py", "SETUP_PROFILES", 120)

    return f"""You are a quantitative trading systems analyst reviewing scanner signal performance.
This is Analysis #{analysis_num} with {len(rows)} fully-scored signals (each scored over a D0..D+3 trading-day window).

## How signals are scored
- bucket=reversal means SHORT thesis (favorable = down); bucket=bounce means LONG (favorable = up).
- recommendation=VETO (from 2026-06-10 on): the signal scored GO/CAUTION but was hard-vetoed by the prior_day_rvol < 1.25 floor you recommended in Analysis #2. These are logged specifically so you can verify the veto: report the vetoed cohort's tradeable_3d rate each cycle, and flag immediately if it drifts toward the GO rate (veto would be discarding edge).
- d0_pct..d3_pct: cumulative close-vs-entry-open raw price move per day.
- mfe_atr_3d / mae_atr_3d: max favorable / adverse excursion over the window in ATRs.
- tradeable_3d: MFE >= 1.0 ATR at any point in the window (primary success metric).
- days_to_1atr: trading days until the 1-ATR favorable target hit ('' = never).
- adverse_before_fav_atr: worst adverse run (ATRs) BEFORE the favorable target hit — this measures how EARLY the scanner fires. Large values on reversals mean the stock kept squeezing up before cracking.
- Pre-trade criterion features logged per signal (values at alert time): reversal signals carry pct_from_9ema, pct_change_3, gap_pct, prior_day_range_atr, prior_day_rvol, premarket_rvol; bounce signals carry selloff_total_pct, pct_off_30d_high, pct_off_52wk_high, pct_change_3, gap_pct, prior_day_range_atr. Use these for criterion-level threshold analysis: compare feature distributions of tradeable vs non-tradeable signals (first-flags only) and look for cut points that would have filtered losers without dropping winners. Blank = the report didn't emit that metric for that signal.
- episode_id / episode_signal_num: repeat flags of the same ticker within {EPISODE_GAP} trading days chain into one episode; their outcome windows OVERLAP, so rows within an episode are NOT independent samples. For any statistical claim, use episode_signal_num == 1 rows (or count distinct episode_id) as the sample. Reprints (episode_signal_num >= 2) may be analyzed separately as a persistence feature — "does a 2nd/3rd consecutive flag predict better odds?" — but never mix them into per-signal rates as if independent.

## Signal outcomes (complete windows only)
{table}

## Current thresholds (read-only context)
Reversal CAP_THRESHOLDS (score >= 4 GO, == 3 CAUTION):
{rev_thresholds}

Bounce SETUP_PROFILES (score >= 5 GO, == 4 CAUTION):
{bounce_thresholds}

## Recommendations from previous analyses (do not repeat unless new data strengthens or reverses them)
{past_text}

## Statistical guardrails — follow strictly
- Do NOT recommend a threshold change based on any cell (bucket x cap x criterion) with n < {MIN_CELL_N}. Say "insufficient sample" instead.
- Prefer pooled-across-cap conclusions at current sample sizes.
- When you cite a rate, include the count (e.g. "4/19"). Note that a 30% vs 50% difference on n<30 is usually noise.
- Distinguish "the scanner is wrong" from "the scanner is early": use days_to_1atr and adverse_before_fav_atr.

## Output format
### PERFORMANCE SUMMARY
### EARLINESS ANALYSIS
Is the reversal scanner early? Quantify using days_to_1atr and adverse_before_fav_atr. Would waiting for a confirmation trigger (or a long-first tactic) have helped, based on this data?
### CRITERIA EFFECTIVENESS
### THRESHOLD RECOMMENDATIONS
For each (only if guardrails allow), one line:
THRESHOLD_CHANGE: <bucket> | <cap or POOLED> | <criterion> | <current> | <recommended> | <evidence with counts>
### ACTION ITEMS
For each actionable follow-up, one line:
ACTION_ITEM: <concise description>"""


def run_claude(prompt: str) -> str:
    """Run claude -p in its own process group so a timeout kills MCP children too."""
    import signal as _signal

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    try:
        proc = subprocess.Popen(
            [CLAUDE_BIN, "-p", prompt, "--permission-mode", "bypassPermissions"],
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
        return (stdout or "").strip()
    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timed out after 30 minutes — killing process group")
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
        except Exception:
            proc.kill()
        return ""


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
