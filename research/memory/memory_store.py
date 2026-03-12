"""Persistent memory store for cross-session learning.

Dual-format: knowledge_base.json (structured facts) + lessons.md (Claude-written reflection).
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from research.experiments.base import ExperimentResult
from research.memory.extractors import (
    extract_failed_experiment,
    extract_feature_importance,
    extract_filter_sweep,
    extract_regime_analysis,
    extract_risk_metrics,
    extract_walk_forward,
)

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _empty_kb() -> Dict:
    """Return an empty knowledge base structure."""
    return {
        "version": 1,
        "last_updated": datetime.now().isoformat(),
        "data_fingerprint": {},
        "baselines": {},
        "feature_rankings": {},
        "regime_findings": [],
        "filter_discoveries": [],
        "failed_experiments": [],
        "walk_forward_stability": {},
        "risk_profiles": {},
        "session_log": [],
    }


class MemoryStore:
    """Persistent memory across research sessions."""

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.kb_path = self.memory_dir / "knowledge_base.json"
        self.lessons_path = self.memory_dir / "lessons.md"
        self.kb: Dict = self.load()

    def load(self) -> Dict:
        """Load knowledge_base.json from disk. Returns empty structure if missing."""
        if self.kb_path.exists():
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                n_sessions = len(data.get("session_log", []))
                logger.info(f"Memory loaded: {n_sessions} prior sessions tracked")
                return data
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Corrupt knowledge_base.json, starting fresh: {e}")
                backup = self.kb_path.with_suffix(".json.bak")
                self.kb_path.rename(backup)
                return _empty_kb()
        else:
            logger.info("No prior memory found, starting fresh")
            return _empty_kb()

    def save(self):
        """Write knowledge_base.json to disk atomically."""
        self.kb["last_updated"] = datetime.now().isoformat()
        tmp = self.kb_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.kb, f, indent=2, cls=_NumpyEncoder)
        tmp.replace(self.kb_path)

    # ------------------------------------------------------------------
    # Baseline & staleness
    # ------------------------------------------------------------------

    def update_baselines(self, config):
        """Re-measure baseline metrics from current CSV data."""
        strategy_csvs = {
            "reversal": (config.reversal_csv, "reversal_open_close_pct", -1),
            "bounce": (config.bounce_csv, "bounce_open_close_pct", 1),
        }

        old_fingerprint = self.kb.get("data_fingerprint", {})
        new_fingerprint = {}

        for strategy in config.strategies:
            if strategy not in strategy_csvs:
                continue
            csv_path, target_col, pnl_sign = strategy_csvs[strategy]
            try:
                df = pd.read_csv(str(csv_path)).dropna(subset=["ticker", "date"])
                pnl = df[target_col] * pnl_sign * 100

                self.kb.setdefault("baselines", {})[strategy] = {
                    "n": len(df),
                    "win_rate": round(float((pnl > 0).mean() * 100), 1),
                    "avg_pnl": round(float(pnl.mean()), 2),
                    "median_pnl": round(float(pnl.median()), 2),
                    "last_measured": datetime.now().strftime("%Y-%m-%d"),
                }

                new_fingerprint[f"{strategy}_n"] = len(df)
                mtime = os.path.getmtime(str(csv_path))
                new_fingerprint[f"{strategy}_csv_mtime"] = datetime.fromtimestamp(mtime).isoformat()

            except Exception as e:
                logger.warning(f"Failed to update baseline for {strategy}: {e}")

        self.kb["data_fingerprint"] = new_fingerprint

    def check_staleness(self) -> List[str]:
        """Compare current data fingerprint against stored findings. Return warnings."""
        warnings = []
        fp = self.kb.get("data_fingerprint", {})

        for strategy in ["reversal", "bounce"]:
            current_n = fp.get(f"{strategy}_n", 0)
            # Check if any findings reference a smaller N
            baseline = self.kb.get("baselines", {}).get(strategy, {})
            old_n = baseline.get("n", current_n)

            if current_n > 0 and old_n > 0 and current_n > old_n * 1.1:
                warnings.append(
                    f"{strategy} data grew from {old_n} to {current_n} trades — "
                    f"prior findings may need re-validation"
                )

        if warnings:
            for w in warnings:
                logger.info(f"Staleness warning: {w}")

        return warnings

    # ------------------------------------------------------------------
    # Per-experiment updates
    # ------------------------------------------------------------------

    def update_from_result(self, result: ExperimentResult, config):
        """Extract structured facts from an experiment result and merge into memory."""
        # Check for failures first
        failed = extract_failed_experiment(result, config.min_sample)
        if failed:
            self._merge_failed(failed, result.timestamp)
            return

        # Type-specific extraction
        if result.experiment_type == "feature_importance":
            data = extract_feature_importance(result)
            if data:
                self._merge_feature_rankings(data)

        elif result.experiment_type == "filter_sweep":
            discoveries = extract_filter_sweep(result)
            if discoveries:
                for d in discoveries:
                    self._merge_filter_discovery(d)

        elif result.experiment_type == "regime_analysis":
            data = extract_regime_analysis(result)
            if data:
                self._merge_regime_finding(data)

        elif result.experiment_type == "walk_forward_sensitivity":
            data = extract_walk_forward(result)
            if data:
                self._merge_walk_forward(data)

        elif result.experiment_type == "risk_metrics":
            data = extract_risk_metrics(result)
            if data:
                self.kb.setdefault("risk_profiles", {})[result.strategy] = data

    def _merge_feature_rankings(self, data: Dict):
        """Merge feature importance results, averaging rho for known features."""
        strategy = data["strategy"]
        target = data.get("target", "pnl")
        new_features = data["features"]

        rankings = self.kb.setdefault("feature_rankings", {})
        strat_rankings = rankings.setdefault(strategy, {})
        existing = strat_rankings.get(target, [])

        # Build lookup by feature name
        by_name = {e["feature"]: e for e in existing}

        for nf in new_features:
            fname = nf["feature"]
            if fname in by_name:
                old = by_name[fname]
                # Weighted average of rho
                old_count = old.get("times_seen", 1)
                new_rho = (old["rho"] * old_count + nf["rho"]) / (old_count + 1)
                old["rho"] = round(new_rho, 4)
                old["times_seen"] = old_count + 1
                old["last_session"] = datetime.now().strftime("%Y-%m-%d_%H-%M")
            else:
                by_name[fname] = {
                    "feature": fname,
                    "rho": round(nf["rho"], 4),
                    "times_seen": 1,
                    "last_session": datetime.now().strftime("%Y-%m-%d_%H-%M"),
                }

        # Sort by abs(rho) descending, keep top 15
        sorted_features = sorted(by_name.values(), key=lambda x: abs(x["rho"]), reverse=True)
        strat_rankings[target] = sorted_features[:15]

    def _merge_filter_discovery(self, discovery: Dict):
        """Merge a filter discovery, incrementing times_confirmed if known."""
        discoveries = self.kb.setdefault("filter_discoveries", [])
        desc = discovery["filter_desc"]
        strategy = discovery["strategy"]

        # Look for existing match
        for existing in discoveries:
            if existing["filter_desc"] == desc and existing["strategy"] == strategy:
                existing["times_confirmed"] = existing.get("times_confirmed", 1) + 1
                existing["improvement_pct"] = round(
                    (existing["improvement_pct"] + discovery["improvement_pct"]) / 2, 2
                )
                existing["last_session"] = datetime.now().strftime("%Y-%m-%d_%H-%M")
                if discovery.get("is_significant"):
                    existing["is_significant"] = True
                return

        # New discovery
        discovery["times_confirmed"] = 1
        discovery["first_session"] = datetime.now().strftime("%Y-%m-%d_%H-%M")
        discovery["last_session"] = discovery["first_session"]
        discoveries.append(discovery)

        # Keep only top 30 by improvement
        discoveries.sort(key=lambda x: x.get("improvement_pct", 0), reverse=True)
        self.kb["filter_discoveries"] = discoveries[:30]

    def _merge_regime_finding(self, finding: Dict):
        """Merge a regime finding, incrementing times_confirmed if known."""
        findings = self.kb.setdefault("regime_findings", [])

        for existing in findings:
            if (existing["strategy"] == finding["strategy"]
                    and existing["regime"] == finding["regime"]
                    and existing["best_bucket"] == finding["best_bucket"]):
                existing["times_confirmed"] = existing.get("times_confirmed", 1) + 1
                existing["improvement_pct"] = round(
                    (existing["improvement_pct"] + finding["improvement_pct"]) / 2, 2
                )
                existing["last_session"] = datetime.now().strftime("%Y-%m-%d_%H-%M")
                if finding.get("p_value") and (existing.get("p_value") is None
                                                or finding["p_value"] < existing["p_value"]):
                    existing["p_value"] = finding["p_value"]
                return

        finding["times_confirmed"] = 1
        finding["first_session"] = datetime.now().strftime("%Y-%m-%d_%H-%M")
        finding["last_session"] = finding["first_session"]
        findings.append(finding)

    def _merge_walk_forward(self, data: Dict):
        """Update walk-forward stability for a strategy."""
        wf = self.kb.setdefault("walk_forward_stability", {})
        wf[data["strategy"]] = {
            "verdict": data["verdict"],
            "avg_oos_wr": data.get("avg_oos_wr"),
            "avg_oos_pnl": data.get("avg_oos_pnl"),
            "n_splits": data.get("n_splits", 0),
            "last_tested": datetime.now().strftime("%Y-%m-%d"),
        }

    def _merge_failed(self, failed: Dict, timestamp: str):
        """Record a failed experiment for avoidance."""
        failures = self.kb.setdefault("failed_experiments", [])

        # Dedup by type + strategy + params_key
        for existing in failures:
            if (existing["experiment_type"] == failed["experiment_type"]
                    and existing["strategy"] == failed["strategy"]
                    and existing.get("params_key") == failed.get("params_key")):
                existing["n_available"] = failed.get("n_available", 0)
                existing["session"] = timestamp
                return

        failed["session"] = timestamp
        failures.append(failed)

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    def record_session(self, session_id: str, stats: Dict):
        """Append session metadata to the session log."""
        log = self.kb.setdefault("session_log", [])
        log.append({
            "session_id": session_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "n_experiments": stats.get("total_experiments", 0),
            "n_significant": stats.get("significant_findings", 0),
            "experiment_types": stats.get("experiment_types_used", []),
            "strategies": stats.get("strategies_tested", []),
            "runtime_seconds": stats.get("runtime_seconds", 0),
        })

    # ------------------------------------------------------------------
    # Lessons (markdown)
    # ------------------------------------------------------------------

    def append_lessons(self, reflection_text: str, session_id: str):
        """Append Claude's reflection to lessons.md. Keep last 5 sessions."""
        if not reflection_text or not reflection_text.strip():
            return

        # Read existing
        existing = ""
        if self.lessons_path.exists():
            existing = self.lessons_path.read_text(encoding="utf-8")

        # Append new section
        new_section = f"\n## Session {session_id}\n{reflection_text.strip()}\n\n---\n"

        if not existing.strip():
            content = "# Research Lessons\n" + new_section
        else:
            content = existing.rstrip() + "\n" + new_section

        # Prune to last 5 sessions
        sections = re.split(r"(?=^## Session )", content, flags=re.MULTILINE)
        header = sections[0] if sections else "# Research Lessons\n"
        session_sections = [s for s in sections[1:] if s.strip()]

        if len(session_sections) > 5:
            # Archive older ones
            archived = session_sections[:-5]
            archive_path = self.memory_dir / "lessons_archive.md"
            archive_text = ""
            if archive_path.exists():
                archive_text = archive_path.read_text(encoding="utf-8")
            archive_text += "\n".join(archived)
            archive_path.write_text(archive_text, encoding="utf-8")

            session_sections = session_sections[-5:]

        content = header + "\n".join(session_sections)
        self.lessons_path.write_text(content, encoding="utf-8")
        logger.info(f"Lessons updated for session {session_id}")

    def _load_recent_lessons(self) -> str:
        """Load the last 3 sessions of lessons for prompt injection."""
        if not self.lessons_path.exists():
            return ""

        content = self.lessons_path.read_text(encoding="utf-8")
        sections = re.split(r"(?=^## Session )", content, flags=re.MULTILINE)
        session_sections = [s.strip() for s in sections[1:] if s.strip()]

        if not session_sections:
            return ""

        # Take last 3
        recent = session_sections[-3:]
        return "\n".join(recent)

    # ------------------------------------------------------------------
    # Render for prompt
    # ------------------------------------------------------------------

    def render_for_prompt(self, max_chars: int = 8000) -> str:
        """Render memory as a compact text block for system prompt injection.

        Priority order: baselines, staleness, top features, confirmed findings,
        failed experiments, prior lessons. Truncates to stay within budget.
        """
        session_count = len(self.kb.get("session_log", []))
        if session_count == 0 and not self.kb.get("feature_rankings"):
            return ""

        lines = []
        last_session = self.kb.get("session_log", [{}])[-1].get("session_id", "unknown") if session_count else "none"
        lines.append(f"({session_count} prior sessions, last: {last_session})")
        lines.append("")

        # Baselines
        baselines = self.kb.get("baselines", {})
        if baselines:
            lines.append("BASELINES (current data):")
            for strat, bl in baselines.items():
                lines.append(
                    f"  {strat}: N={bl.get('n', '?')}, WR={bl.get('win_rate', '?')}%, "
                    f"Avg={bl.get('avg_pnl', '?'):+.2f}%, Median={bl.get('median_pnl', '?'):+.2f}%"
                )
            lines.append("")

        # Feature rankings
        rankings = self.kb.get("feature_rankings", {})
        if rankings:
            lines.append("TOP PREDICTORS (by frequency across sessions):")
            for strat, targets in rankings.items():
                for target, features in targets.items():
                    top5 = features[:5]
                    if top5:
                        feat_strs = [
                            f"{f['feature']} (rho={f['rho']}, {f.get('times_seen', 1)}x)"
                            for f in top5
                        ]
                        lines.append(f"  {strat}/{target}: {', '.join(feat_strs)}")
            lines.append("")

        # Confirmed findings (regime + filter, sorted by times_confirmed)
        all_findings = []

        for rf in self.kb.get("regime_findings", []):
            all_findings.append({
                "times": rf.get("times_confirmed", 1),
                "text": (
                    f"{rf['strategy']}: {rf.get('best_bucket', '?')} vs {rf.get('worst_bucket', '?')} "
                    f"({rf['regime']}) {rf.get('improvement_pct', 0):+.1f}% "
                    f"(p={rf.get('p_value', '?')})"
                ),
            })

        for fd in self.kb.get("filter_discoveries", []):
            if fd.get("improvement_pct", 0) > 0:
                all_findings.append({
                    "times": fd.get("times_confirmed", 1),
                    "text": (
                        f"{fd['strategy']}: {fd['filter_desc']} "
                        f"+{fd['improvement_pct']:.1f}% avg P&L "
                        f"(N={fd.get('n_filtered', '?')}"
                        f"{', sig' if fd.get('is_significant') else ''}"
                        f"{', WF tested' if fd.get('walk_forward_tested') else ''})"
                    ),
                })

        all_findings.sort(key=lambda x: x["times"], reverse=True)
        if all_findings:
            lines.append("CONFIRMED FINDINGS:")
            for f in all_findings[:10]:
                lines.append(f"  [{f['times']}x] {f['text']}")
            lines.append("")

        # Walk-forward status
        wf = self.kb.get("walk_forward_stability", {})
        if wf:
            lines.append("WALK-FORWARD STATUS:")
            for strat, data in wf.items():
                if data:
                    lines.append(
                        f"  {strat}: {data['verdict']} "
                        f"(avg OOS WR={data.get('avg_oos_wr', '?')}%, "
                        f"tested {data.get('last_tested', '?')})"
                    )
                else:
                    lines.append(f"  {strat}: NOT YET TESTED")
            lines.append("")

        # Failed experiments
        failures = self.kb.get("failed_experiments", [])
        if failures:
            lines.append("DO NOT RE-RUN (insufficient data):")
            for fe in failures[:5]:
                lines.append(
                    f"  {fe['experiment_type']} {fe['strategy']} "
                    f"{fe.get('params_key', '')} "
                    f"(N={fe.get('n_available', '?')})"
                )
            lines.append("")

        # Recent lessons
        lessons = self._load_recent_lessons()
        if lessons:
            lines.append("PRIOR LESSONS:")
            # Extract just bullet points, skip headers
            for line in lessons.split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    lines.append(f"  {line}")
            lines.append("")

        rendered = "\n".join(lines)

        # Truncate if over budget
        if len(rendered) > max_chars:
            rendered = rendered[:max_chars - 50] + "\n  [memory truncated]"

        return rendered

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self, max_sessions: int = 20, max_findings: int = 30):
        """Drop old/stale entries from the knowledge base."""
        # Session log
        log = self.kb.get("session_log", [])
        if len(log) > max_sessions:
            self.kb["session_log"] = log[-max_sessions:]

        # Failed experiments — remove if data grew enough
        current_fp = self.kb.get("data_fingerprint", {})
        min_sample = 10
        pruned_failures = []
        for fe in self.kb.get("failed_experiments", []):
            strategy = fe.get("strategy", "")
            current_n = current_fp.get(f"{strategy}_n", 0)
            needed = fe.get("n_available", 0)
            # Keep if data hasn't grown enough to make it viable
            if needed > 0 and current_n > 0 and needed < min_sample:
                pruned_failures.append(fe)
            elif fe.get("failure_reason") != "insufficient_data":
                pruned_failures.append(fe)
        self.kb["failed_experiments"] = pruned_failures

        # Feature rankings — drop single-observation features from old sessions
        session_count = len(self.kb.get("session_log", []))
        for strat_rankings in self.kb.get("feature_rankings", {}).values():
            for target, features in strat_rankings.items():
                strat_rankings[target] = [
                    f for f in features
                    if f.get("times_seen", 1) > 1 or session_count <= 10
                ]

        # Filter discoveries — drop unconfirmed old ones
        self.kb["filter_discoveries"] = [
            d for d in self.kb.get("filter_discoveries", [])
            if d.get("times_confirmed", 1) > 1 or session_count <= 10
        ][:max_findings]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-line summary for logging."""
        n_sessions = len(self.kb.get("session_log", []))
        n_features = sum(
            len(features)
            for targets in self.kb.get("feature_rankings", {}).values()
            for features in targets.values()
        )
        n_findings = len(self.kb.get("regime_findings", [])) + len(self.kb.get("filter_discoveries", []))
        n_failures = len(self.kb.get("failed_experiments", []))
        return (
            f"Memory: {n_sessions} sessions, {n_features} features tracked, "
            f"{n_findings} findings, {n_failures} failed experiments"
        )
