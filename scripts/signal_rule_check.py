"""Monthly bounce rule check: the feedback loop on the signal scorecard's data.

Replaces the daily LLM read (scripts/signal_analysis.py, turned off 2026-11-09). That
read re-ran every morning on data that grows one session a day, so it repeated itself
and took small differences for findings. This is plain arithmetic on a slow cadence,
and it stays quiet unless an answer changes.

The question, for each of the six bounce pre-trade rules: on the same days, did names
that passed the rule bank more R than names that failed it?

  rows      data/signal_outcomes.csv, bounce candidates the priority report scored
            (surfaced + its rejected controls; Method's breadth cohort is left out). One row
            per ticker-day; a name in both cohorts counts once, as surfaced.
  R         exit_r: realised R under the live rules (2-min break entry, LOD stop, 1.5-ATR
            armed trail, one re-entry; scripts/backfill_entry_r.py). A day the rules never
            entered is 0R. Rows the replay hasn't reached are left out. Every candidate day
            counts, not only a move's first flag: the outcome is that day's trade, so
            reprints don't share an outcome window the way the old 3-day measure did.
  pass/fail the logged decision when the row has one (decision_evidence, since 10/21);
            older rows are re-scored from their stored values with today's thresholds and
            count only when both setup profiles (weak / strong stock) agree.
  compare   per day with at least one passer and one failer: passers' average R minus
            failers' average R. Days weigh equally. The 95% range comes from resampling
            the days in blocks of five consecutive days, since neighbouring days share a
            market.
  verdict   helps (whole range above zero), hurts (whole range below zero), unclear, or
            too few days (under MIN_DAYS days with both sides).

The same comparison is run on the score (4+/6 against 3 or less, 5+/6 against 4/6), and
the breadth bands are read from Method's edge table (data/bounce_day_edge.json, rebuilt
nightly), whose own drift check is reported as is.

It writes data/signal_rule_check/<date>.md and state.json. It emails only when a verdict
moves into or out of helps/hurts, when a first run finds a rule that hurts, or when the
breadth drift flag first appears. Rule changes still take a review and the trader's OK.

Usage:
    python -m scripts.signal_rule_check                  # check, save, email on a change
    python -m scripts.signal_rule_check --no-email       # check and save only
    python -m scripts.signal_rule_check --out /tmp/rc --no-email   # dry run elsewhere
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("signal_rule_check")

DATA_DIR = PROJECT_ROOT / "data"
OUTCOMES = DATA_DIR / "signal_outcomes.csv"
EDGE = DATA_DIR / "bounce_day_edge.json"
OUT_DIR = DATA_DIR / "signal_rule_check"
EMAIL_TO = "zmburr@gmail.com"

MIN_DAYS = 10          # days with both a passer and a failer before a verdict
N_BOOT = 5000
BLOCK = 5              # consecutive days resampled together
SEED = 7

RULES = (              # (key, label, operator) as BouncePretrade.validate checks them
    ("selloff_total_pct", "Deep selloff", "lte"),
    ("pct_off_30d_high", "Off the 30-day high", "lte"),
    ("gap_pct", "Gap down", "lte"),
    ("prior_day_range_atr", "Prior-day range of 1+ ATR", "gte"),
    ("pct_change_3", "3-day drop", "lte"),
    ("pct_off_52wk_high", "Off the 52-week high", "lte"),
)
SCORE_CHECKS = (       # (key, label, passes, fails) on the n/6 score
    ("score_4plus", "Score 4+/6 vs 3 or less", lambda s: s >= 4, lambda s: s <= 3),
    ("score_5plus", "Score 5+/6 vs 4/6", lambda s: s >= 5, lambda s: s == 4),
)
DECISIVE = ("helps", "hurts")


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def load_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Bounce candidates with a realised R, one row per ticker-day."""
    b = df[(df["bucket"] == "bounce") & df["cohort"].isin(["surfaced", "control"])].copy()
    signals = pd.Series(pd.to_numeric(b["entry_signals"], errors="coerce"), index=b.index)
    r = pd.Series(pd.to_numeric(b["exit_r"], errors="coerce"), index=b.index)
    r = r.where(signals != 0, 0.0)                     # the rules never entered: 0R
    b["r"] = r
    b = b[signals.notna() & b["r"].notna()]
    b["date"] = b["target_date"].astype(str)
    b["_surfaced"] = (b["cohort"] == "surfaced").astype(int)
    b = (b.sort_values("_surfaced", ascending=False)
          .drop_duplicates(["date", "ticker"])
          .drop(columns="_surfaced"))
    b["score_n"] = pd.to_numeric(b["score"].astype(str).str.split("/").str[0], errors="coerce")
    return b.reset_index(drop=True)


def _thresholds(cap: str) -> list[dict]:
    from analyzers.bounce_scorer import SETUP_PROFILES
    return [{k: prof.get_threshold(k, cap) for k, _, _ in RULES} for prof in SETUP_PROFILES.values()]


def _passes(value: float, threshold: float, op: str) -> bool:
    return value <= threshold if op == "lte" else value >= threshold


def rule_state(row: pd.Series, key: str, op: str, thresholds=None) -> bool | None:
    """True / False when the rule's outcome is known for this row, else None."""
    thresholds = thresholds or _thresholds
    ev = row.get("decision_evidence")
    if isinstance(ev, str) and ev.strip():
        try:
            for c in json.loads(ev).get("criteria", []):
                if c.get("key") == key:
                    return bool(c["passed"]) if c.get("input_present", True) else None
        except (ValueError, TypeError, KeyError):
            pass
    value = pd.to_numeric(row.get(key), errors="coerce")
    if pd.isna(value):
        return None
    cap = row.get("cap") if isinstance(row.get("cap"), str) else "Medium"
    states = {_passes(float(value), t[key], op) for t in thresholds(cap)}
    return states.pop() if len(states) == 1 else None    # profiles disagree: ambiguous


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def block_interval(diffs, n_boot: int = N_BOOT, block: int = BLOCK, seed: int = SEED) -> tuple[float, float]:
    """95% range of the mean by a circular moving-block bootstrap over days (in date order)."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    b = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n / b))))
    idx = ((starts[:, :, None] + np.arange(b)) % n).reshape(n_boot, -1)[:, :n]
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def verdict(days: int, lo: float | None, hi: float | None, min_days: int = MIN_DAYS) -> str:
    if days < min_days or lo is None:
        return "too few days"
    return "helps" if lo > 0 else "hurts" if hi < 0 else "unclear"


def compare(rows: pd.DataFrame, passed: pd.Series, min_days: int = MIN_DAYS) -> dict:
    """passed: True / False / None per row. Same-day passers minus failers, in R a name."""
    known = rows.assign(_p=passed.values)[passed.notna().values]
    days, p_means, f_means = [], [], []
    for date, g in known.groupby("date", sort=True):
        p, f = g.loc[g["_p"] == True, "r"], g.loc[g["_p"] == False, "r"]  # noqa: E712
        if len(p) and len(f):
            days.append(date)
            p_means.append(p.mean())
            f_means.append(f.mean())
    out = {"days": len(days), "passers": int((known["_p"] == True).sum()),  # noqa: E712
           "failers": int((known["_p"] == False).sum()), "pass_r": None, "fail_r": None,  # noqa: E712
           "diff": None, "lo": None, "hi": None}
    if days:
        diffs = np.array(p_means) - np.array(f_means)
        out.update(pass_r=float(np.mean(p_means)), fail_r=float(np.mean(f_means)), diff=float(diffs.mean()))
        if len(days) >= 2:
            out["lo"], out["hi"] = block_interval(diffs)
    out["verdict"] = verdict(out["days"], out["lo"], out["hi"], min_days)
    return out


def run_checks(rows: pd.DataFrame, min_days: int = MIN_DAYS, thresholds=None) -> dict:
    thresholds = thresholds or _thresholds
    results = {}
    for key, label, op in RULES:
        state = rows.apply(lambda r: rule_state(r, key, op, thresholds), axis=1)
        results[key] = {"label": label, **compare(rows, state.astype(object), min_days)}
    for key, label, is_pass, is_fail in SCORE_CHECKS:
        s = rows["score_n"]
        state = pd.Series([True if is_pass(v) else False if is_fail(v) else None
                           for v in s], index=rows.index, dtype=object).where(s.notna(), None)
        results[key] = {"label": label, **compare(rows, state, min_days)}
    return results


# ---------------------------------------------------------------------------
# Breadth (read from Method's edge table)
# ---------------------------------------------------------------------------

def breadth_summary(edge: dict | None) -> dict:
    if not edge:
        return {"bands": {}, "drift": None, "built": None}
    bands = {}
    for band, cell in (edge.get("bands") or {}).items():
        since = cell.get("since_study") or {}
        bands[band] = {"study_days": cell.get("days"), "study_r": cell.get("mean_r"),
                       "since_days": since.get("days"), "since_r": since.get("mean_r"),
                       "since_pos_days": since.get("pos_days"), "since": since.get("since")}
    return {"bands": bands, "drift": edge.get("drift"), "built": edge.get("built")}


# ---------------------------------------------------------------------------
# Alerts, report, email
# ---------------------------------------------------------------------------

def alerts(prev: dict | None, results: dict, drift, prev_drift) -> list[str]:
    """Lines worth an email: a verdict moving into or out of helps/hurts (on a first run,
    only rules that hurt), and the breadth drift flag first appearing."""
    lines = []
    for key, res in results.items():
        cur = res["verdict"]
        if prev is None:                      # first run: a baseline, unless a rule hurts
            if cur == "hurts":
                lines.append(f"{res['label']}: hurts ({_fmt_cell(res)})")
            continue
        old = prev.get(key, "unclear")
        if cur != old and (cur in DECISIVE or old in DECISIVE):
            lines.append(f"{res['label']}: {old} -> {cur} ({_fmt_cell(res)})")
    if drift and not prev_drift:
        lines.append(f"Breadth days since the study: {drift.get('mean_r', 0):+.2f}R a name over "
                     f"{drift.get('days')} days, against {drift.get('history_mean_r', 0):+.2f}R in the study")
    return lines


def _r(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}R"


def _fmt_cell(res: dict) -> str:
    if res["diff"] is None:
        return f"{res['days']} days with both sides"
    rng = f", range {res['lo']:+.2f} to {res['hi']:+.2f}" if res["lo"] is not None else ""
    return f"{res['diff']:+.2f}R a name on {res['days']} days{rng}"


def render(results: dict, breadth: dict, rows: pd.DataFrame, today: str, changes: list[str]) -> str:
    dates = sorted(rows["date"].unique())
    span = f"{dates[0]} to {dates[-1]}" if dates else "no data"
    out = [f"# Bounce rule check, {today}", "",
           f"{len(rows):,} bounce candidates with a realised R on {len(dates)} days ({span}). "
           f"R is banked R under the live rules; a day the rules didn't enter is 0R.", ""]
    out += ["## Changes since the last check", ""]
    out += [f"- {c}" for c in changes] if changes else ["- None."]
    out += ["", "## Rules", "",
            "Same-day comparison: passers' average R minus failers' average R, days weighted equally.", "",
            "| Rule | Days with both | Passers | Failers | Difference | 95% range | Verdict |",
            "|---|---:|---:|---:|---:|---|---|"]
    def row(res):
        rng = f"{res['lo']:+.2f} to {res['hi']:+.2f}" if res["lo"] is not None else "—"
        return (f"| {res['label']} | {res['days']} | {_r(res['pass_r'])} ({res['passers']}) | "
                f"{_r(res['fail_r'])} ({res['failers']}) | {_r(res['diff'])} | {rng} | {res['verdict']} |")
    out += [row(results[k]) for k, _, _ in RULES]
    out += ["", "## Score", "",
            "| Check | Days with both | Higher score | Lower score | Difference | 95% range | Verdict |",
            "|---|---:|---:|---:|---:|---|---|"]
    out += [row(results[k]) for k, *_ in SCORE_CHECKS]
    out += ["", "## Breadth (Method's edge table)", ""]
    if breadth["bands"]:
        out += ["| Band | Study days | Study R a name | Days since | R a name since | Green days since |",
                "|---|---:|---:|---:|---:|---:|"]
        for band, c in breadth["bands"].items():
            out.append(f"| {band} | {c['study_days']} | {_r(c['study_r'])} | {c['since_days']} | "
                       f"{_r(c['since_r'])} | {c['since_pos_days']} |")
        out += ["", f"Drift flag: {'ON: ' + json.dumps(breadth['drift']) if breadth['drift'] else 'off'} "
                    f"(table built {breadth['built']})."]
    else:
        out.append("Edge table not found.")
    out += ["", "## How to read this", "",
            f"- A verdict needs {MIN_DAYS}+ days with both a passer and a failer. The range resamples days "
            f"in blocks of {BLOCK}; helps or hurts means the whole range is on one side of zero.",
            "- Rows before 10/21 are re-scored with today's thresholds and count only when the weak- and "
            "strong-stock profiles agree; later rows use the decision the report logged.",
            "- This compares names, not causes: passers and failers differ on the other rules too. "
            "A verdict is a reason to review a rule, not to change it.", ""]
    return "\n".join(out)


def _email(lines: list[str], report_path: Path, today: str) -> bool:
    from support.config import send_email
    items = "".join(f"<li>{line}</li>" for line in lines)
    html = (f"<p>The monthly bounce rule check found {len(lines)} change{'s' if len(lines) != 1 else ''}:</p>"
            f"<ul>{items}</ul><p>Full report on the mini: {report_path}</p>")
    subject = f"Bounce rule check {today}: {len(lines)} change{'s' if len(lines) != 1 else ''}"
    return bool(send_email(EMAIL_TO, subject, html, is_html=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check(outcomes: Path = OUTCOMES, edge_path: Path = EDGE, out_dir: Path = OUT_DIR,
          today: str | None = None, email: bool = True, send=_email) -> dict:
    today = today or datetime.date.today().isoformat()
    rows = load_rows(pd.read_csv(outcomes, low_memory=False))
    results = run_checks(rows)
    edge = json.loads(edge_path.read_text()) if edge_path.exists() else None
    breadth = breadth_summary(edge)

    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    prev = json.loads(state_path.read_text()) if state_path.exists() else None
    changes = alerts(prev and prev.get("verdicts"), results, breadth["drift"], prev and prev.get("drift"))

    report_path = out_dir / f"{today}.md"
    report_path.write_text(render(results, breadth, rows, today, changes), encoding="utf-8")
    sent = None
    if changes and email:
        sent = send(changes, report_path, today)
        logger.info(f"email {'sent' if sent else 'FAILED'}: {len(changes)} change(s)")
    else:
        logger.info(f"no email ({len(changes)} change(s){', email off' if not email else ''})")
    state_path.write_text(json.dumps({
        "run": today, "rows": len(rows), "days": int(rows["date"].nunique()),
        "verdicts": {k: v["verdict"] for k, v in results.items()},
        "results": results, "drift": breadth["drift"]}, indent=1, default=float))
    logger.info(f"wrote {report_path}")
    return {"results": results, "changes": changes, "report": report_path, "sent": sent}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outcomes", type=Path, default=OUTCOMES)
    ap.add_argument("--edge", type=Path, default=EDGE)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--today")
    ap.add_argument("--no-email", action="store_true")
    a = ap.parse_args()
    check(a.outcomes, a.edge, a.out, a.today, email=not a.no_email)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
