"""Claude CLI wrapper — the LLM brain that proposes experiments and synthesizes findings."""

import json
import logging
import os
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a quantitative trading research assistant analyzing backtesting data.

OBJECTIVE: Maximize expectancy (average P&L per trade) for two strategies:
1. REVERSAL (parabolic shorts): ~200+ trades. Negative return = profit (shorts). Measured by -reversal_open_close_pct.
2. BOUNCE (capitulation longs): ~83 trades. Positive return = profit (longs). Measured by bounce_open_close_pct.

CRITICAL — PRE-TRADE vs OUTCOME COLUMNS:
The trader's decision point is PREMARKET on the trade date. Only features available BEFORE entry
are useful for building predictive filters. NEVER propose experiments using outcome columns as
predictive features — they create look-ahead bias and produce useless findings.

OUTCOME columns (NOT available at decision time — only use as targets, never as filters/features):
  trade_grade, bp, npl, size,
  reversal_open_close_pct, reversal_open_low_pct, reversal_open_post_low_pct,
  reversal_open_to_day_after_open_pct, reversal_duration, time_of_reversal,
  bounce_open_close_pct, bounce_open_high_pct, bounce_open_low_pct,
  bounce_open_to_day_after_open_pct, bounce_duration, time_of_bounce,
  day_of_range_pct, close_at_lows, close_at_highs, close_green_red,
  close_above_prior_close, hit_green_red, hit_prior_day_hilo, move_together,
  time_of_high_price, time_of_high_bucket, time_of_low, time_of_low_price,
  time_of_low_bucket, high_to_low_duration_min,
  vol_on_breakout_day, percent_of_vol_on_breakout_day, spy_open_close_pct

PRE-TRADE columns (available premarket — USE THESE for filters/features):
  gap_pct, gap_from_pm_high/low, pct_change_3/15/30/90/120,
  pct_from_9ema/10mav/20mav/50mav/200mav, atr_distance_from_50mav,
  atr_pct, atr_pct_move, avg_daily_vol, consecutive_up/down_days,
  upper/lower_band_distance, bollinger_width, closed_outside_upper/lower_band,
  prior_day_close_vs_high/low_pct, prior_day_range_atr,
  one/two/three_day_before_range_pct, percent_of_premarket_vol, premarket_vol,
  rvol_score, spy_5day_return, uvxy_close,
  selloff_total_pct, pct_off_30d_high, pct_off_52wk_high,
  breaks_ath, breaks_fifty_two_wk, near_52wk_low, breaks_52wk_low

EARLY-SESSION columns (available minutes into session — valid if entry is after that window):
  vol_in_first_5/10/15/30_min, percent_of_vol_in_first_5/10/15/30_min, vol_ratio_5min_to_pm

RULES:
- Statistical rigor: don't trust findings with N < 10. Multiple comparison problem is real.
- Always distinguish in-sample vs out-of-sample findings.
- Cap stratification matters (Micro/Small/Medium/Large/ETF have different dynamics).
- Prefer experiments that VALIDATE prior findings over pure exploration.
- When proposing follow-ups, explain WHY based on what you learned.
- Don't re-run experiments that have already been done (check session history).
- Return JSON only — no markdown, no explanation outside the JSON.

TRADE GRADES: A+, A (best), B, C, F. Focus on A and B grades.
"""


def ask_claude(prompt: str, system: str = None, model: str = "opus",
               timeout: int = 120) -> str:
    """Call Claude via the claude CLI subprocess."""
    cmd = ["claude", "-p", prompt, "--output-format", "text", "--model", model]
    if system:
        cmd.extend(["--system-prompt", system])

    # Remove CLAUDECODE env var to allow nested subprocess calls
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    logger.debug(f"Claude CLI call ({len(prompt)} chars)")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", env=env,
        )
        if result.returncode != 0:
            logger.error(f"Claude CLI error: {result.stderr[:500]}")
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.error(f"Claude CLI timed out after {timeout}s")
        return ""
    except Exception as e:
        logger.error(f"Claude CLI failed: {e}")
        return ""


def _parse_json(text: str) -> Optional[any]:
    """Extract JSON from Claude's response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array or object
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            if start == -1:
                continue
            # Find matching close bracket
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except json.JSONDecodeError:
                            break
        logger.warning(f"Failed to parse JSON from Claude response: {text[:200]}")
        return None


class ClaudeResearcher:
    """LLM brain that proposes experiments and synthesizes findings."""

    def __init__(self, model: str = "opus", timeout: int = 120,
                 memory_context: str = ""):
        self.model = model
        self.timeout = timeout
        self.call_count = 0
        if memory_context:
            self.system_prompt = SYSTEM_PROMPT + "\n\n" + memory_context
        else:
            self.system_prompt = SYSTEM_PROMPT

    def propose_initial_experiments(self, data_overview: str,
                                     experiment_catalog: str) -> List[Dict]:
        """Cold start: propose 2-3 experiments based on data overview."""
        prompt = f"""Here is an overview of the trading data available for research:

{data_overview}

Here are the experiments you can run:

{experiment_catalog}

Propose 2-3 experiments to start with. For each, return JSON:
[
  {{
    "experiment_type": "<name from catalog>",
    "parameters": {{...}},
    "rationale": "<why this experiment first>",
    "priority": 1
  }}
]

Pick experiments that will give broad insight into both strategies. Start with feature_importance or filter_sweep for the strategy with more data, then branch out."""

        response = ask_claude(prompt, system=self.system_prompt, model=self.model,
                              timeout=self.timeout)
        self.call_count += 1

        proposals = _parse_json(response)
        if proposals is None:
            logger.warning("Claude failed to return valid proposals, using defaults")
            return self._default_initial_experiments()

        if isinstance(proposals, dict):
            proposals = [proposals]

        return proposals

    def propose_followups(self, latest_result_summary: str,
                          all_results_summary: str,
                          experiment_catalog: str,
                          completed_experiments: List[str]) -> List[Dict]:
        """After each experiment, propose 0-2 follow-up experiments."""
        prompt = f"""Latest experiment result:
{latest_result_summary}

All results so far:
{all_results_summary}

Already completed experiment types+strategies (do NOT repeat):
{json.dumps(completed_experiments)}

Available experiments:
{experiment_catalog}

Based on these results, propose 0-2 follow-up experiments. Return JSON array (empty [] if no follow-ups needed):
[
  {{
    "experiment_type": "<name>",
    "parameters": {{...}},
    "rationale": "<why this follows from the results>",
    "priority": 1
  }}
]

Prioritize:
1. Validating significant findings (e.g., if a filter improved expectancy, test it walk-forward)
2. Drilling deeper into promising signals (e.g., if a feature is important, sweep its thresholds)
3. Cross-strategy insights (if something works for reversal, test it for bounce)
4. Risk assessment (if expectancy improved, check if risk metrics are acceptable)"""

        response = ask_claude(prompt, system=self.system_prompt, model=self.model,
                              timeout=self.timeout)
        self.call_count += 1

        proposals = _parse_json(response)
        if proposals is None:
            return []
        if isinstance(proposals, dict):
            proposals = [proposals]

        return proposals[:2]  # Cap at 2

    def synthesize_findings(self, all_results_summary: str,
                             session_stats: Dict) -> str:
        """End of session: write the final synthesis for the morning report."""
        prompt = f"""Research session complete. Here are all the results:

{all_results_summary}

Session stats: {json.dumps(session_stats)}

Write a synthesis with:
1. EXECUTIVE SUMMARY: 3-5 bullet points of the most actionable findings for improving expectancy.
2. KEY TAKEAWAYS: For each significant finding, explain what it means for trading decisions.
3. RECOMMENDED ACTIONS: Concrete steps the trader should take (e.g., add a filter, adjust position sizing).
4. NEXT SESSION IDEAS: 2-3 experiments worth running next time.
5. LESSONS LEARNED: 3-5 bullet points reflecting on what you learned this session that should guide future research. Include: what experiment strategies worked/didn't, which hypotheses were confirmed/rejected, what to explore vs avoid next time, and any meta-insights about the data or methodology.

Write in plain English for a trader. Focus on expectancy improvement.
Return as plain text (not JSON)."""

        response = ask_claude(prompt, system=self.system_prompt, model=self.model,
                              timeout=self.timeout)
        self.call_count += 1

        if not response:
            return "Claude synthesis unavailable — see individual experiment results below."

        return response

    def _default_initial_experiments(self) -> List[Dict]:
        """Fallback if Claude fails to propose."""
        return [
            {
                "experiment_type": "feature_importance",
                "parameters": {"strategy": "reversal", "target": "pnl"},
                "rationale": "Identify which features best predict reversal P&L",
                "priority": 1,
            },
            {
                "experiment_type": "filter_sweep",
                "parameters": {"strategy": "reversal", "grade": "A"},
                "rationale": "Find filter thresholds that improve Grade A reversal expectancy",
                "priority": 2,
            },
            {
                "experiment_type": "feature_importance",
                "parameters": {"strategy": "bounce", "target": "pnl"},
                "rationale": "Identify which features best predict bounce P&L",
                "priority": 3,
            },
        ]
