"""Tests for the monthly bounce rule check (scripts/signal_rule_check.py).

Expected values are worked out by hand from the synthetic rows, not snapshots.
"""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import signal_rule_check as rc


# Two fixed profiles so the tests don't depend on today's thresholds.
WEAK = {"selloff_total_pct": -0.30, "pct_off_30d_high": -0.50, "gap_pct": -0.05,
        "prior_day_range_atr": 1.0, "pct_change_3": -0.20, "pct_off_52wk_high": -0.50}
STRONG = dict(WEAK, gap_pct=-0.03)


def thresholds(cap):
    return [WEAK, STRONG]


def _row(date, ticker, r, cohort="control", signals=1, score="3/6", **kw):
    row = {"bucket": "bounce", "cohort": cohort, "target_date": date, "ticker": ticker,
           "cap": "Medium", "score": score, "entry_signals": signals, "exit_r": r,
           "decision_evidence": None}
    for key, _, _ in rc.RULES:
        row[key] = np.nan
    row.update(kw)
    return row


def _days(n, start="2026-07-01"):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


# --- rows -------------------------------------------------------------------

def test_load_rows_counts_no_entry_as_zero_and_drops_unreplayed():
    df = pd.DataFrame([
        _row("2026-07-01", "AAA", np.nan, signals=0),       # rules never entered -> 0R
        _row("2026-07-01", "BBB", np.nan, signals=np.nan),  # replay hasn't run -> dropped
        _row("2026-07-01", "CCC", np.nan, signals=1),       # entered but no exit R -> dropped
        _row("2026-07-01", "DDD", 1.5),
    ])
    rows = rc.load_rows(df)
    assert sorted(rows.ticker) == ["AAA", "DDD"]
    assert rows.set_index("ticker").r.to_dict() == {"AAA": 0.0, "DDD": 1.5}


def test_load_rows_keeps_surfaced_copy_and_skips_breadth_cohort():
    df = pd.DataFrame([
        _row("2026-07-01", "AAA", -1.0, cohort="control"),
        _row("2026-07-01", "AAA", 2.0, cohort="surfaced"),
        _row("2026-07-01", "BBB", 3.0, cohort="breadth"),
    ])
    rows = rc.load_rows(df)
    assert len(rows) == 1 and rows.iloc[0].cohort == "surfaced" and rows.iloc[0].r == 2.0


# --- pass / fail --------------------------------------------------------------

def test_rule_state_uses_logged_decision_first():
    ev = json.dumps({"criteria": [{"key": "gap_pct", "passed": True, "input_present": True},
                                  {"key": "pct_change_3", "passed": False, "input_present": False}]})
    row = pd.Series(_row("2026-10-22", "AAA", 1.0, decision_evidence=ev, gap_pct=0.10))
    assert rc.rule_state(row, "gap_pct", "lte", thresholds) is True     # logged, not re-scored
    assert rc.rule_state(row, "pct_change_3", "lte", thresholds) is None  # input was missing


def test_rule_state_rescores_old_rows_only_when_profiles_agree():
    def state(gap):
        return rc.rule_state(pd.Series(_row("2026-07-01", "A", 0, gap_pct=gap)), "gap_pct", "lte", thresholds)
    assert state(-0.06) is True      # passes -0.05 and -0.03
    assert state(-0.01) is False     # fails both
    assert state(-0.04) is None      # passes strong (-0.03), fails weak (-0.05)
    assert state(np.nan) is None


# --- comparison ----------------------------------------------------------------

def _rows_from(pairs):
    """pairs: list of (date, passed, r)."""
    return pd.DataFrame([{"date": d, "r": r, "_passed": p} for d, p, r in pairs])


def test_compare_weighs_days_equally():
    rows = _rows_from([
        ("2026-07-01", True, 3.0), ("2026-07-01", False, 1.0), ("2026-07-01", False, 1.0),  # +2
        ("2026-07-02", True, 0.0), ("2026-07-02", True, 0.0), ("2026-07-02", False, 1.0),   # -1
        ("2026-07-03", True, 5.0),                                                          # no failer: skipped
    ])
    res = rc.compare(rows, rows["_passed"].astype(object), min_days=2)
    assert res["days"] == 2
    assert res["diff"] == pytest.approx(0.5)
    assert res["pass_r"] == pytest.approx(1.5) and res["fail_r"] == pytest.approx(1.0)
    assert res["passers"] == 4 and res["failers"] == 3


def test_compare_verdicts():
    days = _days(20)
    helps = _rows_from([(d, True, 2.0 + (i % 3) * 0.1) for i, d in enumerate(days)]
                       + [(d, False, -1.0) for d in days])
    hurts = _rows_from([(d, True, -1.0) for d in days] + [(d, False, 1.0) for d in days])
    noise = _rows_from([(d, True, 1.0 if i % 2 else -1.0) for i, d in enumerate(days)]
                       + [(d, False, 0.0) for d in days])
    few = _rows_from([(d, True, 2.0) for d in days[:5]] + [(d, False, -1.0) for d in days[:5]])
    v = {name: rc.compare(x, x["_passed"].astype(object))["verdict"]
         for name, x in {"helps": helps, "hurts": hurts, "noise": noise, "few": few}.items()}
    assert v == {"helps": "helps", "hurts": "hurts", "noise": "unclear", "few": "too few days"}


def test_compare_ignores_rows_with_unknown_state():
    days = _days(12)
    rows = _rows_from([(d, True, 1.0) for d in days] + [(d, False, 0.0) for d in days]
                      + [(d, None, 50.0) for d in days])
    res = rc.compare(rows, rows["_passed"].astype(object))
    assert res["diff"] == pytest.approx(1.0) and res["passers"] == 12 and res["failers"] == 12


def test_block_interval_is_deterministic_and_brackets_the_mean():
    d = np.linspace(-1, 3, 30)
    a, b = rc.block_interval(d), rc.block_interval(d)
    assert a == b
    assert a[0] < d.mean() < a[1]


def test_run_checks_scores_by_n_of_6():
    days = _days(12)
    df = pd.DataFrame([_row(d, f"H{i}", 2.0, score="5/6") for i, d in enumerate(days)]
                      + [_row(d, f"M{i}", 0.0, score="4/6") for i, d in enumerate(days)]
                      + [_row(d, f"L{i}", -1.0, score="2/6") for i, d in enumerate(days)])
    res = rc.run_checks(rc.load_rows(df), thresholds=thresholds)
    assert res["score_5plus"]["diff"] == pytest.approx(2.0) and res["score_5plus"]["verdict"] == "helps"
    # 4+/6 averages (2 + 0) / 2 = 1.0 a day against -1.0
    assert res["score_4plus"]["diff"] == pytest.approx(2.0)


# --- alerts --------------------------------------------------------------------

def _res(verdict, label="Gap down"):
    return {"label": label, "verdict": verdict, "diff": 0.5, "days": 12, "lo": 0.1, "hi": 0.9}


def test_first_run_emails_only_rules_that_hurt():
    lines = rc.alerts(None, {"a": _res("helps"), "b": _res("hurts", "3-day drop"), "c": _res("unclear")},
                      None, None)
    assert len(lines) == 1 and lines[0].startswith("3-day drop: hurts")


def test_alerts_only_on_moves_into_or_out_of_a_decision():
    prev = {"a": "unclear", "b": "helps", "c": "too few days", "d": "hurts"}
    cur = {"a": _res("helps"), "b": _res("unclear"), "c": _res("unclear"), "d": _res("hurts")}
    lines = rc.alerts(prev, cur, None, None)
    assert [ln.split(":")[1].strip().split(" (")[0] for ln in lines] == ["unclear -> helps", "helps -> unclear"]


def test_breadth_drift_alerts_once():
    drift = {"days": 12, "mean_r": -0.4, "history_mean_r": 1.2}
    assert len(rc.alerts({}, {}, drift, None)) == 1
    assert rc.alerts({}, {}, drift, drift) == []


# --- end to end ------------------------------------------------------------------

def test_check_writes_report_and_state_and_emails_on_change(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "_thresholds", thresholds)
    days = _days(12)
    df = pd.DataFrame([_row(d, f"H{i}", -2.0, gap_pct=-0.10, score="5/6") for i, d in enumerate(days)]
                      + [_row(d, f"L{i}", 1.0, gap_pct=0.0, score="4/6") for i, d in enumerate(days)])
    outcomes = tmp_path / "outcomes.csv"
    df.to_csv(outcomes, index=False)
    sent = []
    send = lambda lines, path, today: sent.append(lines) or True  # noqa: E731

    out = rc.check(outcomes, tmp_path / "missing.json", tmp_path / "out", "2026-12-07", send=send)
    assert out["results"]["gap_pct"]["verdict"] == "hurts"
    assert sent and any(ln.startswith("Gap down: hurts") for ln in sent[0])
    state = json.loads((tmp_path / "out" / "state.json").read_text())
    assert state["verdicts"]["gap_pct"] == "hurts"
    report = (tmp_path / "out" / "2026-12-07.md").read_text(encoding="utf-8")
    assert "| Gap down | 12 |" in report and "Edge table not found." in report

    sent.clear()                                         # same data next month: no news
    rc.check(outcomes, tmp_path / "missing.json", tmp_path / "out", "2027-01-04", send=send)
    assert sent == []


def test_check_respects_no_email(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "_thresholds", thresholds)
    days = _days(12)
    df = pd.DataFrame([_row(d, f"H{i}", -2.0, gap_pct=-0.10) for i, d in enumerate(days)]
                      + [_row(d, f"L{i}", 1.0, gap_pct=0.0) for i, d in enumerate(days)])
    df.to_csv(tmp_path / "o.csv", index=False)
    called = []
    out = rc.check(tmp_path / "o.csv", tmp_path / "x.json", tmp_path / "out", "2026-12-07",
                   email=False, send=lambda *a: called.append(a))
    assert out["changes"] and called == [] and out["sent"] is None
