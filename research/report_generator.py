"""Report generator — markdown report + email summary."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from research.config import ResearchConfig
from research.experiments.base import ExperimentResult

logger = logging.getLogger(__name__)


def generate_markdown_report(
    results: List[ExperimentResult],
    synthesis: str,
    session_stats: Dict,
    config: ResearchConfig,
) -> str:
    """Build the full markdown morning report."""

    date_str = datetime.now().strftime("%B %d, %Y")
    sig_results = [r for r in results if r.is_significant]
    insig_results = [r for r in results if not r.is_significant and "FAILED" not in r.summary]

    lines = [
        f"# Overnight Research Report — {date_str}",
        "",
    ]

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(synthesis)
    lines.append("")

    # Significant Findings
    lines.append("## Significant Findings")
    lines.append("")
    if sig_results:
        for r in sig_results:
            lines.append(f"### {r.experiment_type} ({r.strategy})")
            lines.append("")
            lines.append(f"**Finding:** {r.summary}")
            lines.append("")
            if r.statistical_tests:
                lines.append("**Statistical evidence:**")
                for k, v in r.statistical_tests.items():
                    if isinstance(v, (list, dict)):
                        continue
                    lines.append(f"  - {k}: {v}")
                lines.append("")
            if r.metrics:
                # Show key metrics compactly
                key_metrics = {k: v for k, v in r.metrics.items()
                               if not isinstance(v, (list, dict)) and k != "baseline"}
                if key_metrics:
                    lines.append("**Key metrics:**")
                    for k, v in list(key_metrics.items())[:8]:
                        lines.append(f"  - {k}: {v}")
                    lines.append("")
    else:
        lines.append("No statistically significant findings this session.")
        lines.append("")

    # Interesting but Inconclusive
    lines.append("## Interesting but Inconclusive")
    lines.append("")
    noteworthy = [r for r in insig_results
                  if r.metrics and r.summary and "Insufficient" not in r.summary]
    if noteworthy:
        for r in noteworthy[:5]:
            lines.append(f"- **{r.experiment_type}** ({r.strategy}): {r.summary}")
        lines.append("")
    else:
        lines.append("All non-significant results were inconclusive.")
        lines.append("")

    # Experiment Log
    lines.append("## Experiment Log")
    lines.append("")
    lines.append("| # | Type | Strategy | Key Finding | Sig? | Parent |")
    lines.append("|---|------|----------|-------------|------|--------|")
    for i, r in enumerate(results):
        sig_mark = "YES" if r.is_significant else "no"
        parent = r.parent_id[:6] if r.parent_id else "-"
        short_summary = r.summary[:60] + "..." if len(r.summary) > 60 else r.summary
        lines.append(f"| {i+1} | {r.experiment_type} | {r.strategy} | {short_summary} | {sig_mark} | {parent} |")
    lines.append("")

    # Session Stats
    lines.append("## Session Stats")
    lines.append("")
    lines.append(f"- **Duration**: {session_stats.get('runtime_seconds', 0):.0f} seconds")
    lines.append(f"- **Experiments run**: {session_stats.get('total_experiments', 0)}")
    lines.append(f"- **Significant findings**: {session_stats.get('significant_findings', 0)}")
    lines.append(f"- **Claude calls**: {session_stats.get('claude_calls', 0)}")
    lines.append(f"- **Strategies tested**: {', '.join(session_stats.get('strategies_tested', []))}")
    lines.append(f"- **Experiment types used**: {', '.join(session_stats.get('experiment_types_used', []))}")
    lines.append("")

    return "\n".join(lines)


def generate_email_body(
    results: List[ExperimentResult],
    synthesis: str,
) -> str:
    """Build a trimmed email body (executive summary + significant findings only)."""

    date_str = datetime.now().strftime("%B %d, %Y")
    sig_results = [r for r in results if r.is_significant]

    lines = [
        f"Overnight Research Report — {date_str}",
        "",
        "EXECUTIVE SUMMARY",
        "=" * 40,
        synthesis,
        "",
    ]

    if sig_results:
        lines.append("SIGNIFICANT FINDINGS")
        lines.append("=" * 40)
        for r in sig_results:
            lines.append(f"\n[{r.experiment_type}] {r.strategy}")
            lines.append(r.summary)
            if r.statistical_tests:
                key_stats = {k: v for k, v in r.statistical_tests.items()
                             if not isinstance(v, (list, dict))}
                if key_stats:
                    lines.append(f"  Stats: {key_stats}")
        lines.append("")
    else:
        lines.append("No statistically significant findings this session.")
        lines.append("")

    n_total = len(results)
    n_sig = len(sig_results)
    lines.append(f"Session: {n_total} experiments, {n_sig} significant")
    lines.append("")
    lines.append("Full report saved to research/reports/")

    return "\n".join(lines)


def save_and_send_report(
    results: List[ExperimentResult],
    synthesis: str,
    session_stats: Dict,
    config: ResearchConfig,
):
    """Save markdown report to disk and send email summary."""
    # Generate reports
    md_report = generate_markdown_report(results, synthesis, session_stats, config)
    email_body = generate_email_body(results, synthesis)

    # Save markdown
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_path = config.reports_dir / f"report_{timestamp}.md"
    report_path.write_text(md_report, encoding="utf-8")
    logger.info(f"Report saved to {report_path}")

    # Send email
    try:
        from support.config import send_email
        subject = f"Overnight Research — {datetime.now().strftime('%b %d')}"
        n_sig = sum(1 for r in results if r.is_significant)
        if n_sig > 0:
            subject += f" — {n_sig} significant finding{'s' if n_sig > 1 else ''}"

        send_email(
            to_email=config.email_to,
            subject=subject,
            body=email_body,
            attachments=[str(report_path)],
        )
        logger.info(f"Email sent to {config.email_to}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

    return report_path
